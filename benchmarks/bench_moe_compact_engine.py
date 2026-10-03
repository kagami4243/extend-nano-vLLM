"""Real Qwen3-MoE startup and warm Graph generation on spare GPUs."""

import argparse
import json
import multiprocessing as mp
import socket
import traceback
from statistics import median
from time import perf_counter


MODEL = "./models/Qwen3-30B-A3B-Base"


def worker(rank, port, args, queue):
    from nanovllm import LLM, SamplingParams

    llm = None
    try:
        started = perf_counter()
        llm = LLM(
            args.model, data_parallel_size=args.dp, data_parallel_rank=rank,
            tensor_parallel_size=args.tp, enable_expert_parallel=True,
            master_port=port, run_id=f"compact_{port}_dp{rank}",
            moe_dispatch_backend=args.backend if args.dp > 1 else "replicated",
            enforce_eager=False, enable_prefix_caching=True,
            max_num_seqs=max(args.batches), max_num_batched_tokens=4096,
            max_model_len=32, gpu_memory_utilization=args.gpu_memory_utilization,
        )
        init_seconds = perf_counter() - started
        sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=3)
        results = []
        for batch in args.batches:
            prompts = [[1000 + (position + request + rank) % 13 for position in range(8)]
                       for request in range(batch)]
            for _ in range(args.warmup):
                llm.generate(prompts, sampling, use_tqdm=False)
            samples = []
            before = llm.model_runner.decode_graph_replay_count
            for _ in range(args.runs):
                started = perf_counter()
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                samples.append((perf_counter() - started) * 1000)
                assert len(outputs) == batch
                assert all(len(item["token_ids"]) == 3 for item in outputs)
            replays = llm.model_runner.decode_graph_replay_count - before
            assert replays == 2 * args.runs, (batch, replays)
            results.append({"batch": batch, "samples_ms": samples,
                            "median_ms": median(samples), "decode_replays": replays,
                            "first_outputs": [item["token_ids"] for item in outputs[:4]]})
        diagnostics = llm.model_runner.call("get_diagnostics")
        queue.put({"rank": rank, "init_seconds": init_seconds, "batches": results,
                   "graph_batch_sizes": diagnostics[0]["decode_graph_batch_sizes"],
                   "parameter_bytes": [item["parameter_bytes"] for item in diagnostics],
                   "decode_replays": [item["decode_graph_replay_count"] for item in diagnostics]})
    except Exception:
        queue.put({"rank": rank, "error": traceback.format_exc()})
    finally:
        if llm is not None:
            llm.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--backend", choices=("allgather_reduce", "allgather_reducescatter"),
                        default="allgather_reduce")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 17, 512])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--result-file")
    args = parser.parse_args()
    if (min(args.dp, args.tp, args.runs) < 1 or args.warmup < 1
            or any(not 1 <= batch <= 512 for batch in args.batches)):
        parser.error("positive DP/TP/runs/warmup and batch sizes in 1-512 are required")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    queue = context.Queue()
    workers = [context.Process(target=worker, args=(rank, port, args, queue))
               for rank in range(args.dp)]
    try:
        for process in workers:
            process.start()
        results = sorted((queue.get(timeout=600) for _ in workers), key=lambda item: item["rank"])
        assert all("error" not in item for item in results), results
        report = {"model": args.model, "dp": args.dp, "tp": args.tp,
                  "ep": args.dp * args.tp,
                  "backend": args.backend if args.dp > 1 else "replicated",
                  "enforce_eager": False, "prefix_caching": True,
                  "max_model_len": 32, "max_num_batched_tokens": 4096,
                  "max_num_seqs": max(args.batches),
                  "gpu_memory_utilization": args.gpu_memory_utilization,
                  "prompt_tokens": 8, "output_tokens": 3,
                  "warmup": args.warmup, "runs": args.runs, "results": results}
        print(json.dumps(report), flush=True)
        if args.result_file:
            with open(args.result_file, "w") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
    finally:
        for process in workers:
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()
                process.join()
        assert all(process.exitcode == 0 for process in workers)


if __name__ == "__main__":
    main()
