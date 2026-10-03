"""Trace one Qwen3-MoE prefill through HF or nano-vLLM for numeric diagnosis."""

import argparse
import json
from types import MethodType

import torch

from benchmarks.bench_moe_ep import make_prompts


def last_token(value, row=-1):
    if isinstance(value, tuple):
        value = value[0]
    return value.reshape(-1, value.shape[-1])[row].detach().float().cpu()


def install_hooks(layers, backend, vectors, row=-1, capture_layer=None,
                  capture_slice=None, capture_qnorm_layer=None,
                  trace_step=None, capture_decode_step=None,
                  capture_attention_layer=None, capture_kv_cache=False,
                  request_index=None, capture_attention_row=None):
    handles = []

    def active():
        return capture_decode_step is None or trace_step[0] == capture_decode_step

    def capture(key, value, layer_id):
        if capture_layer is not None and int(layer_id) == capture_layer:
            start, end = capture_slice
            vectors[key] = value.reshape(-1, value.shape[-1])[start:end].detach().float().cpu()

    for layer_id, layer in layers:
        if backend == "nano" and int(layer_id) == capture_attention_layer:
            def record_attention(name, value):
                if not active():
                    return
                if isinstance(value, tuple):
                    for index, item in enumerate(value):
                        if isinstance(item, torch.Tensor):
                            if capture_attention_row is not None and item.ndim:
                                item = item[capture_attention_row]
                            vectors[f"{capture_attention_layer}.{name}.{index}"] = (
                                item.detach().float().cpu()
                            )
                elif isinstance(value, torch.Tensor):
                    if capture_attention_row is not None:
                        value = value[capture_attention_row]
                    vectors[f"{capture_attention_layer}.{name}"] = (
                        value.detach().float().cpu()
                    )

            for name, module in (
                ("input_norm", layer.input_layernorm),
                ("qkv", layer.self_attn.qkv_proj),
                ("q_norm", layer.self_attn.q_norm),
                ("k_norm", layer.self_attn.k_norm),
                ("attention", layer.self_attn.attn),
                ("o_proj", layer.self_attn.o_proj),
                ("self_attn", layer.self_attn),
                ("post_attn_norm", layer.post_attention_layernorm),
            ):
                def before_attention(module, inputs, name=name):
                    if not active():
                        return
                    record_attention(f"{name}_input", inputs)
                    if name != "attention" or not capture_kv_cache:
                        return
                    from nanovllm.utils.context import get_context

                    context = get_context()
                    length = int(context.context_lens[request_index].item())
                    block_size = module.k_cache.size(1)
                    blocks = context.block_tables[
                        request_index, :(length + block_size - 1) // block_size
                    ].long()
                    for cache_name, cache in (
                        ("k_history", module.k_cache),
                        ("v_history", module.v_cache),
                    ):
                        vectors[f"{capture_attention_layer}.{cache_name}"] = (
                            cache[blocks].reshape(-1, *cache.shape[2:])[:length]
                            .detach().float().cpu()
                        )

                handles.append(module.register_forward_pre_hook(
                    before_attention
                ))
                handles.append(module.register_forward_hook(
                    lambda module, inputs, output, name=name: record_attention(
                        name, output
                    )
                ))
        if backend == "nano" and int(layer_id) == capture_qnorm_layer:
            def before_qnorm(module, inputs, layer_id=layer_id):
                if not active():
                    return
                vectors[f"{layer_id}.qnorm_input_full"] = inputs[0].detach().cpu()
                vectors[f"{layer_id}.qnorm_weight"] = module.weight.detach().cpu()

            handles.append(layer.self_attn.q_norm.register_forward_pre_hook(before_qnorm))

        def after_gate(module, inputs, output, layer_id=layer_id):
            if not active():
                return
            vectors[f"{layer_id}.router_logits"] = last_token(output, row)

        def before_mlp(module, inputs, layer_id=layer_id):
            if not active():
                return
            vectors[f"{layer_id}.mlp_input"] = last_token(inputs[0], row)
            capture(f"{layer_id}.mlp_input_span", inputs[0], layer_id)

        def after_mlp(module, inputs, output, layer_id=layer_id):
            if not active():
                return
            vectors[f"{layer_id}.mlp_output"] = last_token(output, row)
            capture(f"{layer_id}.mlp_output_span", output, layer_id)

        def after_layer(module, inputs, output, layer_id=layer_id):
            if not active():
                return
            if backend == "nano":
                hidden, residual = output
                output = hidden + residual
            vectors[f"{layer_id}.layer_output"] = last_token(output, row)
            capture(f"{layer_id}.layer_output_span", output, layer_id)

        handles.extend((
            layer.mlp.gate.register_forward_hook(after_gate),
            layer.mlp.register_forward_pre_hook(before_mlp),
            layer.mlp.register_forward_hook(after_mlp),
            layer.register_forward_hook(after_layer),
        ))
    return handles


def compare(vectors, reference):
    results = {}
    for key, value in vectors.items():
        expected = reference["vectors"][key]
        if value.shape != expected.shape:
            results[key] = {
                "actual_shape": list(value.shape),
                "reference_shape": list(expected.shape),
                "comparable": False,
            }
            continue
        difference = value - expected
        results[key] = {
            "max_abs": float(difference.abs().max()),
            "mean_abs": float(difference.abs().mean()),
            "rmse": float(difference.square().mean().sqrt()),
            "relative_l2": float(torch.linalg.vector_norm(difference)
                                 / torch.linalg.vector_norm(expected)),
        }
    return results


def bf16_route_combine(self, expert_output, route_rows, num_tokens, top_k):
    padded = torch.cat((
        expert_output, expert_output.new_zeros((1, self.hidden_size)),
    ))
    routes = padded[route_rows.long()].reshape(num_tokens, top_k, self.hidden_size)
    output = expert_output.new_zeros((num_tokens, self.hidden_size))
    for route in range(top_k):
        output.add_(routes[:, route])
    return output


def bf16_expert_combine(self, expert_output, route_rows, num_tokens, top_k):
    order = torch.argsort(self._diagnostic_experts, dim=1, stable=True)
    sorted_rows = torch.gather(route_rows.reshape(num_tokens, top_k), 1, order)
    return bf16_route_combine(self, expert_output, sorted_rows, num_tokens, top_k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("hf", "nano"), required=True)
    parser.add_argument("--model", default="./models/Qwen3-30B-A3B-Base")
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--tp", type=int)
    parser.add_argument("--shard-across-tp", action="store_true")
    parser.add_argument("--dispatch-backend", choices=("replicated", "all_to_all", "all_to_all_reduce"),
                        default="replicated")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--request-index", type=int, default=2)
    parser.add_argument("--token-position", type=int)
    parser.add_argument("--capture-layer", type=int)
    parser.add_argument("--capture-qnorm-layer", type=int)
    parser.add_argument("--capture-all-prompts", action="store_true")
    parser.add_argument("--probe-rank-inputs", action="store_true")
    parser.add_argument("--probe-layer", type=int, default=1)
    parser.add_argument("--probe-stage", choices=(
        "attn_input", "qkv_output", "q_norm_input", "q_norm_output", "q_ready", "k_ready",
        "v_ready", "qkv_ready",
        "flash_output", "attn_output",
        "mlp_input", "mlp_output", "layer_output",
    ),
                        default="mlp_input")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-broadcast-qnorm-input", action="store_true")
    parser.add_argument("--probe-all-layers", action="store_true")
    parser.add_argument("--probe-full-request", action="store_true")
    parser.add_argument("--probe-full-norm-layers", nargs="+", type=int)
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2"),
                        default="sdpa")
    parser.add_argument("--reference-experts", action="store_true")
    parser.add_argument("--bf16-combine", action="store_true")
    parser.add_argument("--expert-order-combine", action="store_true")
    parser.add_argument("--force-deterministic-qk", action="store_true")
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--capture-decode-step", type=int)
    parser.add_argument("--capture-attention-layer", type=int)
    parser.add_argument("--capture-kv-cache", action="store_true")
    parser.add_argument("--all-prompts", action="store_true")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--reference-file")
    args = parser.parse_args()
    if args.tp is None:
        args.tp = args.ep
    if args.shard_across_tp or args.tp != args.ep:
        parser.error("this DP=1 trace requires EP=TP; independent EP axes are removed")
    if args.bf16_combine and args.expert_order_combine:
        parser.error("select only one BF16 combine mode")
    if args.token_position is not None and (
        not args.all_prompts or not 0 <= args.token_position < args.prompt_tokens
    ):
        parser.error("token position requires --all-prompts and a valid prompt offset")
    if args.capture_layer is not None and (
        args.backend != "nano" or not args.all_prompts or not 0 <= args.capture_layer < 48
    ):
        parser.error("layer capture requires nano, --all-prompts, and a layer in [0, 48)")
    if args.capture_all_prompts and args.capture_layer is None:
        parser.error("full-batch capture requires --capture-layer")
    if args.capture_qnorm_layer is not None and (
        args.backend != "nano" or not args.all_prompts
        or not 0 <= args.capture_qnorm_layer < 48
    ):
        parser.error("Q norm input capture requires nano, --all-prompts, and a valid layer")
    if args.probe_rank_inputs and (args.backend != "nano" or not args.all_prompts):
        parser.error("rank input probe requires nano and --all-prompts")
    if args.probe_only and not args.probe_rank_inputs:
        parser.error("--probe-only requires --probe-rank-inputs")
    if args.probe_full_request and (
        not args.probe_rank_inputs or args.probe_all_layers
        or args.probe_full_norm_layers
    ):
        parser.error("--probe-full-request requires a single-stage rank probe")
    if args.probe_broadcast_qnorm_input and (
        not args.probe_rank_inputs or args.probe_stage != "q_norm_output"
    ):
        parser.error("--probe-broadcast-qnorm-input requires Q norm output rank probe")
    if args.probe_all_layers and (
        not args.probe_rank_inputs or args.probe_broadcast_qnorm_input
        or args.probe_full_norm_layers
    ):
        parser.error("--probe-all-layers requires rank probe without Q norm broadcast")
    if args.probe_full_norm_layers and (
        not args.probe_rank_inputs or args.probe_broadcast_qnorm_input
        or any(not 0 <= layer < 48 for layer in args.probe_full_norm_layers)
    ):
        parser.error("--probe-full-norm-layers requires EP rank probe and valid layers")
    if args.backend == "hf" and (args.reference_experts or args.bf16_combine
                                 or args.expert_order_combine
                                 or args.force_deterministic_qk):
        parser.error("expert execution options require nano-vLLM")
    if args.force_deterministic_qk and args.ep != 1:
        parser.error("forced Q/K norm diagnosis requires EP=1")
    if args.capture_decode_step is not None and (
        args.backend != "nano" or not args.all_prompts
        or not 1 <= args.capture_decode_step < args.output_tokens
    ):
        parser.error("decode capture requires nano, --all-prompts and a valid decode step")
    if args.capture_attention_layer is not None and (
        args.backend != "nano" or not args.all_prompts
        or not 0 <= args.capture_attention_layer < 48
    ):
        parser.error("attention capture requires nano, --all-prompts and a valid layer")
    if args.capture_kv_cache and args.capture_attention_layer is None:
        parser.error("KV cache capture requires --capture-attention-layer")

    prompts = make_prompts(args.seed, args.batch_size, args.prompt_tokens)
    prompt = prompts[args.request_index]
    if args.backend == "hf" and args.all_prompts:
        parser.error("HF trace accepts one request at a time")
    vectors = {}
    handles = []
    rank_input_probe = None
    if args.backend == "hf":
        from transformers import Qwen3MoeForCausalLM

        model = Qwen3MoeForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
            attn_implementation=args.attention,
        ).eval().cuda()
        handles = install_hooks(enumerate(model.model.layers), "hf", vectors)
        with torch.inference_mode():
            logits = model(
                torch.tensor([prompt], device="cuda"), use_cache=False
            ).logits[0, -1].float()
        output_id = int(logits.argmax())
        for handle in handles:
            handle.remove()
    else:
        from nanovllm import LLM, SamplingParams

        llm = LLM(
            args.model, tensor_parallel_size=args.ep, pipeline_parallel_size=1,
            enable_expert_parallel=True,
            moe_shard_across_tp=args.shard_across_tp,
            moe_dispatch_backend=args.dispatch_backend,
            max_model_len=args.prompt_tokens + 9,
            max_num_batched_tokens=args.prompt_tokens * args.batch_size,
            max_num_seqs=args.batch_size, gpu_memory_utilization=0.9,
            enable_prefix_caching=False,
            enable_prefill_batching=args.all_prompts,
            enforce_eager=(args.dispatch_backend != "replicated"
                           or args.capture_decode_step is not None),
        )
        try:
            layers = llm.model_runner.model.model.layers.items()
            if args.force_deterministic_qk:
                for layer in llm.model_runner.model.model.layers.values():
                    if layer.self_attn.use_qk_norm:
                        layer.self_attn.q_norm.deterministic_cuda = True
                        layer.self_attn.k_norm.deterministic_cuda = True
            if args.reference_experts:
                if args.ep != 1:
                    raise ValueError("reference expert comparison requires EP=1")
                for layer in llm.model_runner.model.model.layers.values():
                    layer.mlp._can_use_triton_kernel = lambda hidden: False
            if args.bf16_combine or args.expert_order_combine:
                if args.ep != 1 or args.reference_experts:
                    raise ValueError("BF16 combine comparison requires Triton EP=1")
                for layer in llm.model_runner.model.model.layers.values():
                    if args.expert_order_combine:
                        original_route = layer.mlp._route

                        def route_with_record(hidden, module=layer.mlp,
                                              original=original_route):
                            weights, experts = original(hidden)
                            module._diagnostic_experts = experts
                            return weights, experts

                        layer.mlp._route = route_with_record
                    layer.mlp._combine_expert_routes = MethodType(
                        bf16_expert_combine if args.expert_order_combine
                        else bf16_route_combine, layer.mlp
                    )
            trace_step = [0]
            if args.capture_decode_step is not None:
                selected_row = args.request_index
            elif args.all_prompts:
                selected_row = args.request_index * args.prompt_tokens + (
                    args.prompt_tokens - 1 if args.token_position is None
                    else args.token_position
                )
            else:
                selected_row = -1
            capture_slice = (
                0 if args.capture_all_prompts else (
                    args.request_index if args.capture_decode_step is not None
                    else args.request_index * args.prompt_tokens
                ),
                (args.batch_size if args.capture_decode_step is not None
                 else args.batch_size * args.prompt_tokens)
                if args.capture_all_prompts else (
                    args.request_index + 1 if args.capture_decode_step is not None
                    else (args.request_index + 1) * args.prompt_tokens
                ),
            ) if args.capture_layer is not None else None
            if not args.probe_only:
                handles = install_hooks(
                    layers, "nano", vectors, selected_row, args.capture_layer,
                    capture_slice, args.capture_qnorm_layer,
                    trace_step, args.capture_decode_step,
                    args.capture_attention_layer,
                    args.capture_kv_cache, args.request_index,
                    (args.request_index * args.prompt_tokens
                     if args.capture_attention_layer is not None
                     and args.capture_decode_step is None else None),
                )
            captured_logits = []
            original_sample = llm.model_runner.sampler.forward

            def sample(logits, temperatures):
                captured_logits.append(logits.detach().float().cpu())
                result = original_sample(logits, temperatures)
                trace_step[0] += 1
                return result

            llm.model_runner.sampler.forward = sample
            if args.probe_rank_inputs:
                start = args.request_index * args.prompt_tokens
                if args.probe_full_norm_layers:
                    llm.model_runner.call(
                        "set_moe_full_norm_probe", args.probe_full_norm_layers,
                    )
                elif args.probe_all_layers:
                    llm.model_runner.call(
                        "set_moe_layer_rank_probe", selected_row,
                    )
                else:
                    llm.model_runner.call(
                        "set_moe_input_probe", args.probe_layer,
                        (list(range(start, start + args.prompt_tokens))
                         if args.probe_full_request else
                         [start + offset for offset in (0, 40, 197, 286,
                                                         args.prompt_tokens - 1)]),
                        args.probe_stage,
                        args.probe_broadcast_qnorm_input,
                    )
            outputs = llm.generate(
                prompts if args.all_prompts else [prompt],
                SamplingParams(temperature=0, ignore_eos=True,
                               max_tokens=args.output_tokens),
                use_tqdm=False,
            )
            if args.probe_rank_inputs:
                collector = (
                    "collect_moe_full_norm_probe" if args.probe_full_norm_layers
                    else "collect_moe_layer_rank_probe" if args.probe_all_layers
                    else "collect_moe_input_probe"
                )
                rank_input_probe = llm.model_runner.call(collector)
            output_id = [output["token_ids"] if args.capture_decode_step is not None
                         else output["token_ids"][0] for output in outputs]
            if not args.all_prompts:
                output_id = output_id[0]
            logits = captured_logits[args.capture_decode_step or 0][
                args.request_index if args.all_prompts else 0
            ]
        finally:
            for handle in handles:
                handle.remove()
            llm.exit()
    values, ids = logits.topk(5)
    metadata = {
        "backend": args.backend,
        "ep": args.ep,
        "tp": args.tp,
        "moe_shard_across_tp": args.shard_across_tp,
        "dispatch_backend": args.dispatch_backend,
        "model": args.model,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "request_index": args.request_index,
        "token_position": args.token_position,
        "capture_layer": args.capture_layer,
        "capture_all_prompts": args.capture_all_prompts,
        "rank_input_probe": rank_input_probe,
        "probe_layer": args.probe_layer if args.probe_rank_inputs else None,
        "probe_stage": args.probe_stage if args.probe_rank_inputs else None,
        "probe_all_layers": args.probe_all_layers,
        "probe_full_request": args.probe_full_request,
        "probe_full_norm_layers": args.probe_full_norm_layers,
        "attention": args.attention if args.backend == "hf" else "nano_flash_attention_2",
        "reference_experts": args.reference_experts,
        "bf16_combine": args.bf16_combine,
        "expert_order_combine": args.expert_order_combine,
        "force_deterministic_qk": args.force_deterministic_qk,
        "output_tokens": args.output_tokens,
        "capture_decode_step": args.capture_decode_step,
        "capture_attention_layer": args.capture_attention_layer,
        "capture_attention_row": (
            args.request_index * args.prompt_tokens
            if args.capture_attention_layer is not None
            and args.capture_decode_step is None else None
        ),
        "capture_kv_cache": args.capture_kv_cache,
        "prompt": prompts if args.all_prompts else prompt,
        "output_id": output_id,
        "top5_ids": ids.tolist(),
        "top5_logits": values.tolist(),
        "candidate_logits": {str(token): float(logits[token])
                             for token in (76007, 9216, 228, 238)},
    }
    torch.save({"metadata": metadata, "vectors": vectors}, args.output_file)
    result = {"metadata": {key: value for key, value in metadata.items()
                           if key != "prompt"}, "output_file": args.output_file,
              "num_vectors": len(vectors)}
    if args.reference_file:
        reference = torch.load(args.reference_file, map_location="cpu",
                               weights_only=True)
        if reference["metadata"]["prompt"] != metadata["prompt"]:
            raise ValueError("reference prompt differs")
        result["reference_file"] = args.reference_file
        result["differences"] = compare(vectors, reference)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
