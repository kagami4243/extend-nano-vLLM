"""Compare atomic expert accumulation with deterministic route-order combine."""

import argparse
import json
from statistics import median
from time import perf_counter

import torch
import triton
import triton.language as tl


@triton.jit
def route_order_combine(
    expert_output_ptr,
    route_rows_ptr,
    output_ptr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROUND_EACH_ROUTE: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    total = tl.full((BLOCK_H,), 0, tl.float32)
    for route in range(TOP_K):
        row = tl.load(route_rows_ptr + token * TOP_K + route)
        values = tl.load(
            expert_output_ptr + row * HIDDEN + columns,
            mask=(row < NUM_ROWS) & (columns < HIDDEN),
            other=0,
        )
        total += values.to(tl.float32)
        if ROUND_EACH_ROUTE:
            total = total.to(output_ptr.dtype.element_ty).to(tl.float32)
    tl.store(output_ptr + token * HIDDEN + columns, total, columns < HIDDEN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=[16, 1024])
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--result-file")
    args = parser.parse_args()
    torch.manual_seed(17)
    results = []
    for tokens in args.tokens:
        num_routes = tokens * args.top_k
        route_ids = torch.randperm(num_routes, device="cuda", dtype=torch.int32)
        token_ids = (route_ids // args.top_k).long()
        expert_output = torch.randn(
            num_routes, args.hidden_size, device="cuda", dtype=torch.bfloat16
        )

        def atomic():
            output = torch.zeros(
                tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16
            )
            output.index_add_(0, token_ids, expert_output)
            return output

        def ordered(round_each_route):
            route_rows = torch.empty(num_routes, device="cuda", dtype=torch.int32)
            route_rows.scatter_(
                0, route_ids,
                torch.arange(num_routes, device="cuda", dtype=torch.int32),
            )
            output = torch.empty(
                tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16
            )
            route_order_combine[(tokens, triton.cdiv(args.hidden_size, 128))](
                expert_output, route_rows, output,
                HIDDEN=args.hidden_size, TOP_K=args.top_k,
                NUM_ROWS=num_routes, BLOCK_H=128,
                ROUND_EACH_ROUTE=round_each_route,
            )
            return output

        timings = {}
        outputs = {}
        functions = (
            ("atomic", atomic),
            ("ordered", lambda: ordered(False)),
            ("ordered_bf16", lambda: ordered(True)),
        )
        for name, fn in functions:
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            latencies = []
            for _ in range(args.runs):
                start = perf_counter()
                output = fn()
                torch.cuda.synchronize()
                latencies.append((perf_counter() - start) * 1000)
            timings[name] = median(latencies)
            outputs[name] = output
        reference = expert_output.float()[route_ids.argsort().long()]
        reference = reference.reshape(tokens, args.top_k, args.hidden_size).sum(1)
        results.append({
            "tokens": tokens,
            "hidden_size": args.hidden_size,
            "top_k": args.top_k,
            "atomic_ms": timings["atomic"],
            "ordered_ms": timings["ordered"],
            "ordered_bf16_ms": timings["ordered_bf16"],
            "atomic_max_error": float((outputs["atomic"].float() - reference).abs().max()),
            "ordered_max_error": float((outputs["ordered"].float() - reference).abs().max()),
            "ordered_bf16_max_error": float((outputs["ordered_bf16"].float() - reference).abs().max()),
            "ordered_repeat_exact": all(torch.equal(ordered(False), outputs["ordered"]) for _ in range(5)),
            "ordered_bf16_repeat_exact": all(torch.equal(ordered(True), outputs["ordered_bf16"]) for _ in range(5)),
        })
    if args.result_file:
        with open(args.result_file, "w") as file:
            json.dump(results, file, indent=2)
            file.write("\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
