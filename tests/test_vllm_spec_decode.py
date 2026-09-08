# /data0/fwy/Codes/speculative_qwen3_offline.py
from vllm import LLM, SamplingParams

TARGET_MODEL = "/data1/model/qwen/Qwen/Qwen3-8B"
DRAFT_MODEL = "/data0/fwy/Codes/model/Qwen3-8B-speculator.eagle3"

llm = LLM(
    model=TARGET_MODEL,
    tokenizer=TARGET_MODEL,
    dtype="bfloat16",
    trust_remote_code=True,
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    speculative_config={
        "method": "eagle3",
        "model": DRAFT_MODEL,
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
        temperature=0.0,  # EAGLE3 通常用贪婪采样获得较高接受率
        max_tokens=256,
    ),
)

print(outputs[0].outputs[0].text)