import argparse
import time
from random import randint, seed


def main():
    parser = argparse.ArgumentParser(description="Measure a synthetic extend-nano-vLLM workload.")
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    args = parser.parse_args()
    if min(args.num_seqs, args.max_input_len, args.max_output_len) < 1:
        parser.error("all workload dimensions must be positive")

    from nanovllm import LLM, SamplingParams

    seed(0)
    llm = LLM(args.model, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, args.max_output_len))
        for _ in range(args.num_seqs)
    ]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    try:
        llm.generate(["Benchmark: "], SamplingParams())
        t = time.time()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        t = time.time() - t
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / t
        print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
