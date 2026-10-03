import multiprocessing as mp
import socket
from queue import Empty
import traceback


def _run_replica(
    replica_id,
    data_parallel_size,
    model,
    indexed_prompts,
    sampling_params,
    result_queue,
    kwargs,
):
    from nanovllm import LLM

    llm = None
    try:
        replica_kwargs = dict(kwargs)
        master_port = replica_kwargs.get("master_port", 0)
        shared_ep = kwargs.get("enable_expert_parallel", False) and data_parallel_size > 1
        if master_port and not shared_ep:
            replica_kwargs["master_port"] = master_port + replica_id
        run_id = replica_kwargs.get("run_id", "")
        if run_id:
            replica_kwargs["run_id"] = f"{run_id}_dp{replica_id}"
        llm = LLM(
            model,
            data_parallel_size=data_parallel_size,
            data_parallel_rank=replica_id,
            **replica_kwargs,
        )
        indices, prompts = zip(*indexed_prompts)
        outputs = llm.generate(list(prompts), sampling_params, use_tqdm=False)
        result_queue.put((replica_id, list(zip(indices, outputs)), None))
    except Exception:
        result_queue.put((replica_id, [], traceback.format_exc()))
    finally:
        if llm is not None:
            llm.exit()


def generate_data_parallel(
    model,
    prompts,
    sampling_params,
    data_parallel_size=2,
    **kwargs,
):
    """Offline DP; shared EP pads unequal token batches and coordinates completion."""
    if data_parallel_size < 1:
        raise ValueError("data_parallel_size must be positive")
    if data_parallel_size > len(prompts):
        raise ValueError("each data-parallel replica needs at least one prompt")
    if "data_parallel_size" in kwargs or "data_parallel_rank" in kwargs:
        raise ValueError(
            "data-parallel coordinates are managed by "
            "generate_data_parallel"
        )
    if isinstance(sampling_params, list) and len(sampling_params) != len(prompts):
        raise ValueError("sampling_params must cover every prompt")
    kwargs = dict(kwargs)
    if kwargs.get("enable_expert_parallel", False) and data_parallel_size > 1:
        if len(prompts) % data_parallel_size:
            raise ValueError("global EP requires a balanced number of requests per DP rank")
        if any(not isinstance(prompt, list) or not prompt
               or any(type(token) is not int for token in prompt) for prompt in prompts):
            raise ValueError("global EP offline generation requires prompt token IDs")
        parameters = sampling_params if isinstance(sampling_params, list) else [sampling_params]
        if any(not parameter.ignore_eos for parameter in parameters):
            raise ValueError("global EP requires ignore_eos=True for synchronized steps")
        if (len({parameter.max_tokens for parameter in parameters}) != 1
                or any(parameter.max_tokens < 1 for parameter in parameters)):
            raise ValueError("global EP requires the same positive max_tokens on every request")
        if not kwargs.get("master_port", 0):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                kwargs["master_port"] = sock.getsockname()[1]

    assignments = [[] for _ in range(data_parallel_size)]
    for index, prompt in enumerate(prompts):
        assignments[index % data_parallel_size].append((index, prompt))

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for replica_id, indexed_prompts in enumerate(assignments):
        replica_sampling = (
            [sampling_params[index] for index, _ in indexed_prompts]
            if isinstance(sampling_params, list) else sampling_params
        )
        process = ctx.Process(
            target=_run_replica,
            args=(
                replica_id,
                data_parallel_size,
                model,
                indexed_prompts,
                replica_sampling,
                result_queue,
                kwargs,
            ),
        )
        process.start()
        processes.append(process)

    indexed_outputs = []
    replica_ids = []
    errors = []
    for _ in processes:
        try:
            replica_id, outputs, error = result_queue.get(timeout=180)
        except Empty:
            errors.append("timed out waiting for a data-parallel replica")
            break
        replica_ids.append(replica_id)
        indexed_outputs.extend(outputs)
        if error is not None:
            errors.append(error)

    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
            errors.append(f"replica process {process.pid} did not exit")
        elif process.exitcode != 0:
            errors.append(
                f"replica process {process.pid} exited with {process.exitcode}"
            )
    if errors:
        raise RuntimeError("data-parallel generation failed:\n" + "\n".join(errors))

    indexed_outputs.sort(key=lambda item: item[0])
    return [output for _, output in indexed_outputs], sorted(replica_ids)
