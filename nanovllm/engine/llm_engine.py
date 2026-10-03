import atexit
from dataclasses import fields
import socket
from time import perf_counter
import uuid
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.speculative import verify_greedy_proposals


class LLMEngine:

    def __init__(self, model, **kwargs):
        if "moe_global_dp" in kwargs:
            raise ValueError("EP spans DP * TP automatically; remove moe_global_dp")
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        if config.master_port == 0:
            config.master_port = self._find_free_port()
        if not config.run_id:
            config.run_id = uuid.uuid4().hex
        self.config = config
        self._exited = False
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        world_size = config.model_parallel_size
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def exit(self):
        if self._exited:
            return
        self._exited = True
        try:
            self.model_runner.call("exit")
        finally:
            del self.model_runner
            for process in self.ps:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join()
            atexit.unregister(self.exit)

    @property
    def cache_stats(self):
        return self.scheduler.block_manager.stats.copy()

    def relocate_experts(self, placement_by_layer: dict[str, list[int]]):
        """Apply a new MoE placement between requests without recapturing graphs."""
        if not self.config.moe_dynamic_placement:
            raise RuntimeError("dynamic expert placement is disabled")
        if not self.scheduler.is_finished():
            raise RuntimeError("engine must be idle with no pending requests")
        hf_config = self.config.hf_config
        expected_layers = {
            str(layer) for layer in range(hf_config.num_hidden_layers)
            if layer not in getattr(hf_config, "mlp_only_layers", [])
            and (layer + 1) % hf_config.decoder_sparse_step == 0
        }
        if not isinstance(placement_by_layer, dict) or set(placement_by_layer) != expected_layers:
            raise ValueError("placement must cover every MoE layer")
        ep_size = self.config.effective_expert_parallel_size
        experts_per_rank = hf_config.num_experts // ep_size
        placements = {}
        for layer, owners in placement_by_layer.items():
            if (not isinstance(owners, (tuple, list))
                    or len(owners) != hf_config.num_experts
                    or any(type(owner) is not int or not 0 <= owner < ep_size
                           for owner in owners)
                    or any(owners.count(rank) != experts_per_rank
                           for rank in range(ep_size))):
                raise ValueError(f"invalid MoE placement for layer {layer}")
            placements[layer] = tuple(owners)
        return self.model_runner.call("relocate_moe_experts", placements)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if self.config.speculative_config is not None:
            if sampling_params.temperature != 0:
                raise ValueError("EAGLE3 speculative decoding currently requires temperature=0")
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if len(prompt) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError(
                "prompt and completion exceed max_model_len: "
                f"{len(prompt)} + {sampling_params.max_tokens} > "
                f"{self.config.max_model_len}"
            )
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        if self.config.moe_global_dp and self.scheduler.is_finished():
            self.model_runner.call("run", [], True)
            return [], 0
        seqs, is_prefill = self.scheduler.schedule()
        scheduled_token_count = sum(seq.num_scheduled_tokens for seq in seqs)
        if self.config.speculative_config is not None and not is_prefill:
            # EAGLE's draft forward writes proposal KV before target
            # verification, so reserve the full proposal range first.
            for seq in seqs:
                self.scheduler.reserve_speculation(seq)
            proposals = self.model_runner.call("propose_eagle3_batch", seqs)
            base_num_tokens = []
            for seq, proposal_tokens in zip(seqs, proposals):
                base = self.scheduler.begin_speculation(seq, proposal_tokens)
                base_num_tokens.append(base)
                # The current sequence tail is the previous target output. It
                # has not entered the target KV cache yet, so verify it with
                # the drafts.
                seq.num_computed_tokens = base - 1

            target_tokens, target_aux_hidden_states = self.model_runner.call(
                "verify_eagle3_batch",
                seqs,
                [len(proposal_tokens) + 1 for proposal_tokens in proposals],
            )
            for seq, proposal_tokens, base, tokens, aux in zip(
                seqs, proposals, base_num_tokens, target_tokens,
                target_aux_hidden_states,
            ):
                if len(tokens) != len(proposal_tokens) + 1:
                    raise RuntimeError(
                        "EAGLE3 target verification returned an incomplete batch: "
                        f"proposals={len(proposal_tokens)}, "
                        f"target_tokens={len(tokens)}"
                    )
                verification = verify_greedy_proposals(
                    proposal_tokens, tokens
                )
                self.scheduler.postprocess_speculation(
                    seq,
                    base,
                    verification.accepted_tokens,
                    verification.replacement_token,
                )
                committed_tail = seq.token_ids[base:]
                if committed_tail and not seq.is_finished:
                    self.model_runner.call(
                        "commit_eagle3",
                        seq,
                        committed_tail,
                        aux[:len(committed_tail)],
                        base - 1,
                    )
                if seq.is_finished:
                    self.model_runner.call("release_eagle3_state", seq.seq_id)
            token_ids = []
            scheduled_token_count = sum(
                len(seq) - base for seq, base in zip(seqs, base_num_tokens)
            )
        else:
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        if self.config.speculative_config is not None and not is_prefill:
            num_tokens = -scheduled_token_count
        else:
            num_tokens = scheduled_token_count if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        local_finished = self.scheduler.is_finished()
        if self.config.moe_global_dp:
            return not self.model_runner.call("has_unfinished_dp", not local_finished)
        return local_finished

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
