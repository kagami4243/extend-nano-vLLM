"""Asymmetric prefix-cache hits with four-GPU DP2 x TP2 = EP4."""

import argparse
import json
import multiprocessing as mp
import socket
import traceback
from types import MethodType


MODEL = "./models/Qwen3-30B-A3B-Base"


def worker(rank, port, backend, result_queue):
    from nanovllm import LLM, SamplingParams
    from nanovllm.utils.context import get_context

    llm = None
    try:
        llm = LLM(
            MODEL, data_parallel_size=2, data_parallel_rank=rank,
            tensor_parallel_size=2, enable_expert_parallel=True,
            moe_dispatch_backend=backend, master_port=port,
            run_id=f"prefix_ep_{port}_dp{rank}", enable_prefix_caching=True,
            enforce_eager=False, moe_prefill_piece=True,
            moe_prefill_piece_capture_sizes=(256, 512),
            max_model_len=528, max_num_batched_tokens=512, max_num_seqs=1,
            gpu_memory_utilization=0.85,
        )
        events = []
        coordinate = llm.model_runner.coordinate_ep_batch

        def record_batch(runner, count, is_prefill, device):
            coordinate(count, is_prefill, device)
            context = get_context()
            events.append({"tokens": count, "prefill": is_prefill,
                           "counts": context.moe_token_counts,
                           "eager": context.force_eager})

        llm.model_runner.coordinate_ep_batch = MethodType(record_batch, llm.model_runner)
        sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=4)
        prefix = [1000 + position % 13 for position in range(256)]
        warm_prompt = prefix if rank == 0 else [1290 + position % 7 for position in range(256)]
        llm.generate([warm_prompt], sampling, use_tqdm=False)

        def execute(prompt):
            before = llm.cache_stats["reused_tokens"]
            events.clear()
            output = llm.generate([prompt], sampling, use_tqdm=False)[0]["token_ids"]
            return {"reused_tokens": llm.cache_stats["reused_tokens"] - before,
                    "tokens": output, "events": list(events)}

        prompt = ([1000 + position % 13 for position in range(512)] if rank == 0 else
                  [1100 + position % 11 for position in range(512)])
        hot = execute(prompt)
        assert hot["reused_tokens"] == (256 if rank == 0 else 0), hot
        assert any(event["counts"] == (256, 512) and event["eager"]
                   for event in hot["events"]), hot

        # Invalidate cache lookup only; both replicas now recompute the same requests.
        llm.scheduler.block_manager.hash_to_block_id.clear()
        cold = execute(prompt)
        assert cold["reused_tokens"] == 0, cold
        assert hot["tokens"] == cold["tokens"], {"hot": hot, "cold": cold}
        assert len(hot["tokens"]) == 4
        assert all(len(tokens) == 4 for tokens in (hot["tokens"], cold["tokens"]))

        llm.scheduler.max_num_batched_tokens = 256
        chunk_prompt = (prefix + [1200 + position % 17 for position in range(256)]
                        if rank == 0 else [1400 + position % 19 for position in range(512)])
        chunked = execute(chunk_prompt)
        assert chunked["reused_tokens"] == (256 if rank == 0 else 0), chunked
        assert len(chunked["tokens"]) == 4
        assert any(event["counts"] in ((1, 256), (0, 1)) and event["eager"]
                   for event in chunked["events"]), chunked
        diagnostics = llm.model_runner.call("get_diagnostics")
        assert all(item["decode_graph_replay_count"] > 0 for item in diagnostics)
        result_queue.put({"rank": rank, "backend": backend, "hot": hot,
                          "cold": cold, "chunked": chunked,
                          "decode_graph_replays": [item["decode_graph_replay_count"]
                                                   for item in diagnostics]})
    except Exception:
        result_queue.put({"rank": rank, "error": traceback.format_exc()})
    finally:
        if llm is not None:
            llm.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("allgather_reduce", "allgather_reducescatter"),
                        default="allgather_reduce")
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    results = context.Queue()
    workers = [context.Process(target=worker, args=(rank, port, args.backend, results))
               for rank in range(2)]
    try:
        for process in workers:
            process.start()
        actual = sorted((results.get(timeout=600) for _ in workers), key=lambda item: item["rank"])
        print(json.dumps(actual), flush=True)
        assert all("error" not in item for item in actual), actual
    finally:
        for process in workers:
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()
                process.join()
        assert all(process.exitcode == 0 for process in workers)
    print("asymmetric DP prefix-cache integration passed", flush=True)


if __name__ == "__main__":
    main()
