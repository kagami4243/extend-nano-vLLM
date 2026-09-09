import argparse

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with nano-vLLM.")
    parser.add_argument("model", help="Local Hugging Face model directory")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    try:
        outputs = llm.generate(prompts, sampling_params)
        for prompt, output in zip(prompts, outputs):
            print("\n")
            print(f"Prompt: {prompt!r}")
            print(f"Completion: {output['text']!r}")
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
