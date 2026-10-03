"""HF Qwen3-MoE next-token logits for a selected benchmark prompt."""

import argparse
import json

import torch
from transformers import Qwen3MoeForCausalLM

from benchmarks.bench_moe_ep import make_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-30B-A3B-Base")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--request-index", type=int, default=2)
    parser.add_argument("--attention", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--generate-tokens", type=int, default=1)
    parser.add_argument("--all-requests", action="store_true")
    parser.add_argument("--result-file")
    args = parser.parse_args()
    if args.generate_tokens < 1:
        parser.error("generate-tokens must be positive")
    prompts = make_prompts(args.seed, args.batch_size, args.prompt_tokens)
    if not args.all_requests and not 0 <= args.request_index < len(prompts):
        parser.error("request index is outside batch")

    model = Qwen3MoeForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        attn_implementation=args.attention,
    ).eval().cuda()
    input_ids = torch.tensor(
        prompts if args.all_requests else [prompts[args.request_index]],
        device="cuda", dtype=torch.int64,
    )
    runs = []
    with torch.inference_mode():
        for _ in range(args.runs):
            current_ids = input_ids
            if args.all_requests:
                past_key_values = None
                generated = []
                top2_by_step = []
                for _ in range(args.generate_tokens):
                    result = model(
                        current_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    past_key_values = result.past_key_values
                    logits = result.logits[:, -1].float()
                    values, ids = logits.topk(2, dim=-1)
                    token = logits.argmax(dim=-1)
                    generated.append(token.cpu().tolist())
                    top2_by_step.append({
                        "ids": ids.cpu().tolist(),
                        "logits": values.cpu().tolist(),
                    })
                    current_ids = token[:, None]
                runs.append({
                    "generated": [list(row) for row in zip(*generated)],
                    "top2_by_step": top2_by_step,
                })
                continue
            generated = []
            for _ in range(args.generate_tokens):
                logits = model(current_ids, use_cache=False).logits[0, -1].float()
                token = int(logits.argmax())
                generated.append(token)
                current_ids = torch.cat((
                    current_ids,
                    torch.tensor([[token]], device="cuda", dtype=torch.int64),
                ), dim=1)
            values, ids = logits.topk(10)
            runs.append({
                "generated": generated,
                "top10_ids": ids.cpu().tolist(),
                "top10_logits": values.cpu().tolist(),
                "candidate_logits": {
                    str(token): float(logits[token]) for token in (76007, 9216)
                },
            })
    result = {
        "model": args.model,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "request_index": None if args.all_requests else args.request_index,
        "all_requests": args.all_requests,
        "attention": args.attention,
        "dtype": "bfloat16",
        "generate_tokens": args.generate_tokens,
        "runs": runs,
    }
    if args.result_file:
        with open(args.result_file, "w") as file:
            json.dump(result, file, indent=2)
            file.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
