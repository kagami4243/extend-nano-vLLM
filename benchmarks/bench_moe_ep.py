"""Steady-state Qwen3-MoE benchmark with DP=1 and EP=TP."""

import argparse
import json
import os
from statistics import median
from time import perf_counter
from types import MethodType


def make_prompts(seed, count, length):
    return [
        [1000 + seed * count + request]
        + [100 + (position + request) % 97 for position in range(length - 1)]
        for request in range(count)
    ]


def run(args):
    import torch

    if args.backend == "nano":
        from nanovllm import LLM, SamplingParams

        effective_tp = args.ep
        if args.shard_across_tp or args.tp != args.ep:
            raise ValueError("this DP=1 benchmark requires EP=TP; independent axes are removed")
        extra = {
            "enable_expert_parallel": True,
            "moe_dispatch_backend": args.dispatch_backend,
            "moe_expert_capacity": args.expert_capacity,
            "moe_expert_capacity_factor": args.capacity_factor,
            "moe_expert_placement": args.placement_file,
            "moe_prefill_piece": args.moe_prefill_piece,
        }
        if args.moe_prefill_piece:
            extra["moe_prefill_piece_capture_sizes"] = tuple(
                args.moe_prefill_piece_capture_sizes
                or (args.batch_size * args.prompt_tokens,)
            )
        llm = LLM(
            args.model,
            tensor_parallel_size=effective_tp,
            pipeline_parallel_size=args.pp,
            max_model_len=args.prompt_tokens + args.output_tokens + 8,
            max_num_batched_tokens=args.prompt_tokens * args.batch_size,
            max_num_seqs=args.batch_size,
            enable_prefill_batching=args.enable_prefill_batching,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=not args.disable_prefix_cache,
            enforce_eager=args.enforce_eager,
            **extra,
        )
        if args.eager_op:
            if effective_tp != 1 or args.pp != 1 or args.ep != 1:
                raise ValueError("selective eager ablation currently supports only EP=TP=PP=1")
            from nanovllm.layers.layernorm import RMSNorm
            from nanovllm.layers.rotary_embedding import RotaryEmbedding
            from nanovllm.layers.sampler import Sampler

            model = llm.model_runner.model.model
            if args.eager_op == "qk_norm":
                norms = (
                    norm
                    for layer in model.layers.values()
                    for norm in (layer.self_attn.q_norm, layer.self_attn.k_norm)
                )
                for norm in norms:
                    norm.rms_forward = MethodType(RMSNorm.rms_forward.__wrapped__, norm)
            elif args.eager_op == "residual_norm":
                norms = [model.norm]
                norms.extend(
                    norm for layer in model.layers.values()
                    for norm in (layer.input_layernorm, layer.post_attention_layernorm)
                )
                for norm in norms:
                    norm.rms_forward = MethodType(RMSNorm.rms_forward.__wrapped__, norm)
                    norm.add_rms_forward = MethodType(RMSNorm.add_rms_forward.__wrapped__, norm)
            elif args.eager_op == "rope":
                for layer in model.layers.values():
                    rope = layer.self_attn.rotary_emb
                    rope.forward = MethodType(RotaryEmbedding.forward.__wrapped__, rope)
            elif args.eager_op == "sampler":
                sampler = llm.model_runner.sampler
                sampler.forward = MethodType(Sampler.forward.__wrapped__, sampler)
        sampling = SamplingParams(
            temperature=0, ignore_eos=True, max_tokens=args.output_tokens
        )
        if llm.config.moe_dispatch_backend != args.dispatch_backend:
            raise RuntimeError("requested MoE dispatch backend was not enabled")
        if args.moe_prefill_piece and not llm.model_runner.prefill_piece_enabled:
            raise RuntimeError("requested MoE prefill piece graph was not enabled")

        def generate(prompts):
            return [
                output["token_ids"]
                for output in llm.generate(prompts, sampling, use_tqdm=False)
            ]

    else:
        from vllm import LLM, SamplingParams, TokensPrompt

        if (args.dispatch_backend != "replicated" or args.expert_capacity is not None
                or args.capacity_factor is not None
                or args.placement_file is not None):
            raise ValueError("MoE dispatch, capacity and placement flags control nano-vLLM only")
        llm = LLM(
            model=args.model,
            tokenizer=args.model,
            dtype="bfloat16",
            tensor_parallel_size=args.ep,
            pipeline_parallel_size=args.pp,
            enable_expert_parallel=True,
            max_model_len=args.prompt_tokens + args.output_tokens + 8,
            max_num_batched_tokens=args.prompt_tokens * args.batch_size,
            max_num_seqs=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=not args.disable_prefix_cache,
            enable_chunked_prefill=True,
            enforce_eager=args.enforce_eager,
        )
        sampling = SamplingParams(
            temperature=0, ignore_eos=True, max_tokens=args.output_tokens
        )

        def generate(prompts):
            outputs = llm.generate(
                [TokensPrompt(prompt_token_ids=p) for p in prompts],
                sampling,
                use_tqdm=False,
            )
            return [list(output.outputs[0].token_ids) for output in outputs]

    try:
        for warmup in range(args.warmup_runs):
            generate(make_prompts(1 + warmup, args.batch_size, args.prompt_tokens))
        if args.backend == "nano":
            if (args.moe_prefill_piece
                    and args.batch_size * args.prompt_tokens not in llm.model_runner.prefill_piece_graphs):
                raise RuntimeError("requested MoE prefill shape was not captured")
            diagnostics = llm.model_runner.call("get_diagnostics")
            graph = bool(getattr(llm.model_runner, "graphs", {}))
            piece_graph = llm.model_runner.prefill_piece_enabled
        else:
            diagnostics = None
            graph = not args.enforce_eager
            # The vLLM configuration alone does not show whether this prompt
            # shape actually replayed a prefill graph.
            piece_graph = None
        top2_history = []
        if args.record_top2:
            if args.backend != "nano":
                raise ValueError("top-2 logits observation supports nano-vLLM only")
            original_forward = llm.model_runner.sampler.forward

            def observed_forward(logits, temperatures):
                sampled = original_forward(logits, temperatures)
                values, ids = logits.float().topk(2, dim=-1)
                top2_history.append({
                    "ids": ids.cpu().tolist(),
                    "values": values.cpu().tolist(),
                    "sampled": sampled.cpu().tolist(),
                })
                return sampled

            llm.model_runner.sampler.forward = observed_forward
        latencies = []
        run_diagnostics = []
        first_output = None
        output_history = []
        for repeat in range(args.runs):
            if args.backend == "nano":
                torch.cuda.synchronize()
                cache_before = llm.cache_stats
                prefill_before = llm.scheduler.stats["prefill_steps"]
            start = perf_counter()
            outputs = generate(
                make_prompts(
                    args.repeat_seed if args.repeat_seed is not None else 100 + repeat,
                    args.batch_size, args.prompt_tokens,
                )
            )
            if args.backend == "nano":
                torch.cuda.synchronize()
            latencies.append((perf_counter() - start) * 1000)
            if args.backend == "nano":
                cache_after = llm.cache_stats
                run_diagnostics.append({
                    "prefix_cache_hits": (
                        cache_after["prefix_cache_hits"] - cache_before["prefix_cache_hits"]
                    ),
                    "reused_tokens": (
                        cache_after["reused_tokens"] - cache_before["reused_tokens"]
                    ),
                    "prefill_steps": (
                        llm.scheduler.stats["prefill_steps"] - prefill_before
                    ),
                })
            if first_output is None:
                first_output = outputs
            if args.repeat_seed is not None:
                output_history.append(outputs)
            if len(outputs) != args.batch_size or any(
                len(tokens) != args.output_tokens for tokens in outputs
            ):
                raise RuntimeError("unexpected number of output tokens")
        diagnostics_after = (
            llm.model_runner.call("get_diagnostics")
            if args.backend == "nano" else None
        )
        after_by_rank = (
            {rank["rank"]: rank for rank in diagnostics_after}
            if diagnostics_after is not None else None
        )
        graph_replays = (
            {
                rank["rank"]: after_by_rank[rank["rank"]]["decode_graph_replay_count"]
                - rank["decode_graph_replay_count"]
                for rank in diagnostics
            }
            if args.backend == "nano" else None
        )
        if args.require_graph_replay and (
            graph_replays is None or any(count < 1 for count in graph_replays.values())
        ):
            raise RuntimeError(f"decode CUDA Graph did not replay on every rank: {graph_replays}")
        decode_graph_vs_eager = None
        if args.compare_decode_eager:
            from nanovllm.layers.moe import ExpertParallelMoE
            from nanovllm.utils.context import get_context

            runner = llm.model_runner
            original_run_model = runner.run_model
            moe_layers = [
                module for module in runner.model.modules()
                if isinstance(module, ExpertParallelMoE)
            ]
            records = []

            @torch.inference_mode()
            def compare_run_model(self, input_ids, positions, is_prefill):
                graph_logits = original_run_model(input_ids, positions, is_prefill)
                if is_prefill or records:
                    return graph_logits
                graph_logits = graph_logits.clone()
                graph_hidden = self.graph_vars["outputs"][:input_ids.size(0)].clone()
                context = get_context()
                previous_prefill = context.is_prefill
                previous = [layer.graph_safe_decode for layer in moe_layers]
                try:
                    context.is_prefill = False
                    static_hidden = self.model(input_ids, positions)
                    static_logits = self.model.compute_logits(static_hidden)
                    for layer in moe_layers:
                        layer.graph_safe_decode = False
                    eager_hidden = self.model(input_ids, positions)
                    eager_logits = self.model.compute_logits(eager_hidden)
                finally:
                    context.is_prefill = previous_prefill
                    for layer, value in zip(moe_layers, previous):
                        layer.graph_safe_decode = value

                def tensor_difference(left, right):
                    difference = (left.float() - right.float()).abs()
                    return {
                        "different_elements": int((difference != 0).sum().item()),
                        "total_elements": difference.numel(),
                        "max_abs": float(difference.max().item()),
                        "mean_abs": float(difference.mean().item()),
                    }

                differences = (graph_logits.float() - eager_logits.float()).abs()
                graph_ids = graph_logits.argmax(dim=-1)
                eager_ids = eager_logits.argmax(dim=-1)
                changed = torch.where(graph_ids != eager_ids)[0].tolist()
                records.append({
                    "decode_batch": input_ids.size(0),
                    "direct_attention_is_prefill": False,
                    "different_logit_elements": int((differences != 0).sum().item()),
                    "total_logit_elements": differences.numel(),
                    "max_abs_logit_difference": float(differences.max().item()),
                    "mean_abs_logit_difference": float(differences.mean().item()),
                    "argmax_changed_rows": changed,
                    "graph_vs_static_hidden": tensor_difference(
                        graph_hidden, static_hidden
                    ),
                    "static_vs_eager_hidden": tensor_difference(
                        static_hidden, eager_hidden
                    ),
                    "graph_vs_static_logits": tensor_difference(
                        graph_logits, static_logits
                    ),
                    "static_vs_eager_logits": tensor_difference(
                        static_logits, eager_logits
                    ),
                    "changed_row_top2": [
                        {
                            "row": row,
                            "graph": graph_logits[row].float().topk(2).values.tolist(),
                            "graph_ids": graph_logits[row].float().topk(2).indices.tolist(),
                            "eager": eager_logits[row].float().topk(2).values.tolist(),
                            "eager_ids": eager_logits[row].float().topk(2).indices.tolist(),
                        }
                        for row in changed
                    ],
                })
                return graph_logits

            runner.run_model = MethodType(compare_run_model, runner)
            try:
                generate(make_prompts(
                    args.repeat_seed if args.repeat_seed is not None else 100,
                    args.batch_size, args.prompt_tokens,
                ))
            finally:
                runner.run_model = original_run_model
            decode_graph_vs_eager = records
        result = {
            "backend": args.backend,
            "model": args.model,
            "legacy_ep": args.legacy_ep,
            "moe_shard_across_tp": args.shard_across_tp,
            "tp": effective_tp if args.backend == "nano" else args.ep,
            "pp": args.pp,
            "ep": args.ep,
            "batch_size": args.batch_size,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "max_model_len": args.prompt_tokens + args.output_tokens + 8,
            "max_num_batched_tokens": args.prompt_tokens * args.batch_size,
            "enable_prefill_batching": (
                llm.config.enable_prefill_batching if args.backend == "nano" else True
            ),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "dispatch_backend": (
                args.dispatch_backend if args.backend == "nano" else "vllm_internal"
            ),
            "expert_capacity": args.expert_capacity,
            "expert_capacity_factor": args.capacity_factor,
            "placement_file": args.placement_file,
            "enable_prefix_caching": not args.disable_prefix_cache,
            "repeat_seed": args.repeat_seed,
            "cuda_graph_decode": graph,
            "prefill_piece_enabled": piece_graph,
            "moe_prefill_piece_requested": args.moe_prefill_piece,
            "moe_prefill_piece_capture_sizes": (
                list(llm.config.moe_prefill_piece_capture_sizes)
                if args.backend == "nano" and args.moe_prefill_piece else None
            ),
            "torchdynamo_disabled": os.environ.get("TORCHDYNAMO_DISABLE") == "1",
            "deterministic_qk_norm": (
                os.environ.get("NANOVLLM_EXPERIMENTAL_DETERMINISTIC_QK_NORM") == "1"
            ),
            "fp32_ep_reduce": (
                os.environ.get("NANOVLLM_EXPERIMENTAL_FP32_EP_REDUCE") == "1"
            ),
            "fp64_moe_combine": (
                os.environ.get("NANOVLLM_EXPERIMENTAL_FP64_MOE_COMBINE") == "1"
            ),
            "fp64_staged_reduce": (
                os.environ.get("NANOVLLM_EXPERIMENTAL_FP64_STAGED_REDUCE") == "1"
            ),
            "eager_op": args.eager_op,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "latency_ms": latencies,
            "run_diagnostics": run_diagnostics if args.backend == "nano" else None,
            "median_ms": median(latencies),
            "output_tokens_per_s": (
                args.batch_size * args.output_tokens * 1000 / median(latencies)
            ),
            "first_output": first_output,
            "output_history": output_history if args.repeat_seed is not None else None,
            "top2_history": top2_history if args.record_top2 else None,
            "diagnostics": diagnostics,
            "diagnostics_after": diagnostics_after,
            "graph_batch_sizes": (
                getattr(llm.model_runner, "graph_bs", [])
                if args.backend == "nano" and graph else None
            ),
            "graph_replays_by_rank": graph_replays,
            "decode_graph_vs_eager": decode_graph_vs_eager,
        }
        return result
    finally:
        if args.backend == "nano":
            llm.exit()
        else:
            llm.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("nano", "vllm"), default="nano")
    parser.add_argument("--model", default="./models/Qwen3-30B-A3B-Base")
    parser.add_argument("--tp", type=int)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--legacy-ep", action="store_true")
    parser.add_argument("--shard-across-tp", action="store_true")
    parser.add_argument("--dispatch-backend", choices=("replicated", "all_to_all", "all_to_all_reduce"),
                        default="replicated")
    parser.add_argument("--expert-capacity", type=int)
    parser.add_argument("--capacity-factor", type=float)
    parser.add_argument("--placement-file")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument(
        "--enable-prefill-batching", action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--moe-prefill-piece", action="store_true")
    parser.add_argument("--moe-prefill-piece-capture-sizes", type=int, nargs="+")
    parser.add_argument("--repeat-seed", type=int)
    parser.add_argument("--record-top2", action="store_true")
    parser.add_argument("--require-graph-replay", action="store_true")
    parser.add_argument("--compare-decode-eager", action="store_true")
    parser.add_argument("--eager-op", choices=("qk_norm", "residual_norm", "rope", "sampler"))
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--result-file")
    args = parser.parse_args()
    if args.tp is None:
        args.tp = args.ep
    if min(args.tp, args.pp, args.ep, args.batch_size, args.prompt_tokens,
           args.output_tokens, args.runs, args.warmup_runs) < 1:
        parser.error("parallel sizes and workload dimensions must be positive")
    if args.eager_op and args.backend != "nano":
        parser.error("--eager-op supports nano-vLLM only")
    if args.moe_prefill_piece and args.backend != "nano":
        parser.error("--moe-prefill-piece supports nano-vLLM only")
    if args.moe_prefill_piece_capture_sizes and not args.moe_prefill_piece:
        parser.error("capture sizes require --moe-prefill-piece")
    if args.require_graph_replay and (args.backend != "nano" or args.enforce_eager):
        parser.error("--require-graph-replay requires nano-vLLM graph mode")
    if args.compare_decode_eager and (
        args.backend != "nano" or args.enforce_eager
        or args.ep != 1 or args.tp != 1 or args.pp != 1
    ):
        parser.error("--compare-decode-eager requires nano-vLLM EP=TP=PP=1 graph mode")
    if args.shard_across_tp or args.tp != args.ep:
        parser.error("this DP=1 benchmark requires EP=TP; independent EP axes are removed")
    if args.capacity_factor is not None and args.expert_capacity is not None:
        parser.error("capacity factor and fixed expert capacity are mutually exclusive")
    result = run(args)
    if args.result_file:
        with open(args.result_file, "w") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
