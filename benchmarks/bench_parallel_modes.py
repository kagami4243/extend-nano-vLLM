"""Steady-state DP/PP/TP inference with persistent replica engines.

Example:
  CUDA_VISIBLE_DEVICES=1,2,3,4 python -m benchmarks.bench_parallel_modes \
      --backend nano --mode dp2tp2 --model ./models/Qwen3-8B

Run each mode in a fresh process. Construction and graph warmup are excluded
from measured rounds; DP replicas execute each round concurrently.
"""

import argparse
import json
import multiprocessing as mp
import os
import socket
from queue import Empty
from statistics import median
from time import perf_counter
import traceback


MODES = {
    "single": (1, 1, 1),
    "tp2": (1, 2, 1),
    "dp2": (2, 1, 1),
    "pp2": (1, 1, 2),
    "tp2pp2": (1, 2, 2),
    "dp2tp2": (2, 2, 1),
}


def prompts_for_round(seed: int, prompt_tokens: int, batch_size: int):
    return [
        [1000 + seed * batch_size + request]
        + [100 + (position + request) % 97 for position in range(prompt_tokens - 1)]
        for request in range(batch_size)
    ]


def replica_main(rank, args, commands, responses):
    dp, tp, pp = MODES[args.mode]
    llm = None
    try:
        if args.backend == "vllm":
            visible = args.visible_devices.split(",")
            start = rank * tp * pp
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                visible[start:start + tp * pp]
            )
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
            from vllm import LLM, SamplingParams, TokensPrompt

            llm = LLM(
                model=args.model,
                tokenizer=args.model,
                dtype="bfloat16",
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                max_model_len=args.prompt_tokens + args.output_tokens + 8,
                max_num_batched_tokens=args.prompt_tokens * (args.batch_size // dp),
                max_num_seqs=args.batch_size // dp,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                enable_expert_parallel=args.enable_expert_parallel,
                enforce_eager=False,
            )
            sampling = SamplingParams(
                temperature=0, ignore_eos=True, max_tokens=args.output_tokens
            )

            def generate(prompts):
                results = llm.generate(
                    [TokensPrompt(prompt_token_ids=p) for p in prompts],
                    sampling,
                    use_tqdm=False,
                )
                return [list(item.outputs[0].token_ids) for item in results]

            info = {"enforce_eager": False, "rank": rank}
        else:
            from nanovllm import LLM, SamplingParams

            llm = LLM(
                args.model,
                data_parallel_size=dp,
                data_parallel_rank=rank,
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                master_port=args.ep_master_port,
                run_id=f"parallel_ep_{args.ep_master_port}_{rank}" if args.ep_master_port else "",
                max_model_len=args.prompt_tokens + args.output_tokens + 8,
                max_num_batched_tokens=args.prompt_tokens * (args.batch_size // dp),
                max_num_seqs=args.batch_size // dp,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enable_prefix_caching=True,
                enable_expert_parallel=args.enable_expert_parallel,
                enable_prefill_batching=args.enable_prefill_batching,
                moe_prefill_piece=args.moe_prefill_piece,
                moe_prefill_piece_capture_sizes=(
                    (args.prompt_tokens * (args.batch_size // dp),)
                    if args.moe_prefill_piece else ()
                ),
                enforce_eager=pp > 1,
            )
            sampling = SamplingParams(
                temperature=0, ignore_eos=True, max_tokens=args.output_tokens
            )

            def generate(prompts):
                return [
                    item["token_ids"]
                    for item in llm.generate(prompts, sampling, use_tqdm=False)
                ]

            info = {
                "enforce_eager": pp > 1,
                "rank": rank,
                "enable_prefill_batching": llm.config.enable_prefill_batching,
                "diagnostics": llm.model_runner.call("get_diagnostics"),
            }
        responses.put(("ready", rank, info))
        while True:
            message = commands.get()
            if message is None:
                break
            round_id, indexed_prompts = message
            indices, prompts = zip(*indexed_prompts)
            token_lists = generate(list(prompts))
            if any(len(tokens) != args.output_tokens for tokens in token_lists):
                raise RuntimeError("unexpected output length")
            if args.backend == "nano" and round_id == "warmup":
                info = {
                    "cuda_graph_decode": bool(getattr(llm.model_runner, "graphs", {})),
                    "prefill_piece_enabled": llm.model_runner.prefill_piece_enabled,
                    "reused_tokens": llm.cache_stats["reused_tokens"],
                }
                responses.put((round_id, rank, (list(zip(indices, token_lists)), info)))
            else:
                responses.put((round_id, rank, list(zip(indices, token_lists))))
    except Exception:
        responses.put(("error", rank, traceback.format_exc()))
    finally:
        if llm is not None:
            if args.backend == "nano":
                llm.exit()
            else:
                llm.llm_engine.engine_core.shutdown()


def run(args):
    dp, tp, pp = MODES[args.mode]
    args.ep_master_port = 0
    if args.backend == "nano" and args.enable_expert_parallel and dp > 1:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            args.ep_master_port = sock.getsockname()[1]
    if args.moe_prefill_piece and args.backend != "nano":
        raise ValueError("MoE prefill piece flag requires nano backend")
    if args.batch_size % dp:
        raise ValueError("batch size must be divisible by data parallel size")
    if len(args.visible_devices.split(",")) < dp * tp * pp:
        raise ValueError("not enough visible GPUs")
    ctx = mp.get_context("spawn")
    responses = ctx.Queue()
    command_queues = [ctx.Queue() for _ in range(dp)]
    workers = [
        ctx.Process(target=replica_main, args=(rank, args, command_queues[rank], responses))
        for rank in range(dp)
    ]
    for worker in workers:
        worker.start()

    def receive(expected_round):
        outputs = []
        extra = []
        for _ in workers:
            try:
                round_id, rank, value = responses.get(timeout=args.timeout)
            except Empty as error:
                raise TimeoutError(f"{expected_round} timed out") from error
            if round_id == "error":
                raise RuntimeError(f"replica {rank} failed:\n{value}")
            if round_id != expected_round:
                raise RuntimeError(f"expected {expected_round}, got {round_id}")
            if expected_round == "ready":
                extra.append(value)
            elif expected_round == "warmup" and args.backend == "nano":
                pairs, info = value
                outputs.extend(pairs)
                extra.append(info)
            else:
                outputs.extend(value)
        outputs.sort(key=lambda pair: pair[0])
        return [tokens for _, tokens in outputs], extra

    def execute(round_id, seed):
        prompts = prompts_for_round(seed, args.prompt_tokens, args.batch_size)
        start = perf_counter()
        for rank, command_queue in enumerate(command_queues):
            assignment = [
                (index, prompt)
                for index, prompt in enumerate(prompts)
                if index % dp == rank
            ]
            command_queue.put((round_id, assignment))
        outputs, extra = receive(round_id)
        return (perf_counter() - start) * 1000, outputs, extra

    try:
        _, startup_info = receive("ready")
        warmup_info = []
        for seed in range(1, args.warmup_runs + 1):
            _, _, warmup_info = execute("warmup", seed)
        samples = []
        first_outputs = None
        for repeat in range(args.runs):
            latency, outputs, _ = execute(repeat, 100 + repeat)
            samples.append(latency)
            if first_outputs is None:
                first_outputs = outputs
        return {
            "backend": args.backend,
            "mode": args.mode,
            "model": args.model,
            "dp": dp,
            "tp": tp,
            "pp": pp,
            "batch_size": args.batch_size,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "max_model_len": args.prompt_tokens + args.output_tokens + 8,
            "max_num_batched_tokens_per_replica": args.prompt_tokens * (args.batch_size // dp),
            "max_num_seqs_per_replica": args.batch_size // dp,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_prefix_caching": True,
            "enable_expert_parallel": args.enable_expert_parallel,
            "enable_prefill_batching": (
                startup_info[0]["enable_prefill_batching"]
                if args.backend == "nano" else True
            ),
            "moe_prefill_piece": args.moe_prefill_piece,
            "dp_execution": (
                "synchronous_global_ep" if args.ep_master_port
                else "independent_replica_engines"
            ),
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "latency_ms": samples,
            "latency_ms_median": median(samples),
            "output_tokens_per_s": args.batch_size * args.output_tokens * 1000 / median(samples),
            "first_outputs": first_outputs,
            "startup_info": startup_info,
            "warmup_info": warmup_info,
        }
    finally:
        for command_queue in command_queues:
            command_queue.put(None)
        for worker in workers:
            worker.join(timeout=20)
            if worker.is_alive():
                worker.terminate()
                worker.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("nano", "vllm"), default="nano")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model", default="./models/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--moe-prefill-piece", action="store_true")
    parser.add_argument(
        "--enable-prefill-batching", action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--result-file")
    args = parser.parse_args()
    args.visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    if min(args.batch_size, args.prompt_tokens, args.output_tokens,
           args.runs, args.warmup_runs) < 1:
        parser.error("workload dimensions and runs must be positive")
    result = run(args)
    if args.result_file:
        with open(args.result_file, "w") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
