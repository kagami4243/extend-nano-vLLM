"""Manual vLLM EAGLE3 comparison harness.

This is intentionally a command-line smoke test rather than a pytest test:
it requires a separate vLLM installation and local target/draft checkpoints.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM EAGLE3 as a comparison baseline.")
    parser.add_argument("--target-model", required=True, help="Local target model directory")
    parser.add_argument("--draft-model", required=True, help="Local EAGLE3 draft model directory")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.target_model,
        tokenizer=args.target_model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        speculative_config={
            "method": "eagle3",
            "model": args.draft_model,
            "num_speculative_tokens": 3,
            "max_model_len": 4096,
        },
    )
    tokenizer = llm.get_tokenizer()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "用三句话解释什么是推测解码。"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(
            temperature=0.0,  # EAGLE3 normally benefits from greedy sampling.
            max_tokens=256,
        ),
    )
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
