import multiprocessing as mp
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
        if master_port:
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
    if data_parallel_size < 1:
        raise ValueError("data_parallel_size must be positive")
    if data_parallel_size > len(prompts):
        raise ValueError("each data-parallel replica needs at least one prompt")
    if "data_parallel_size" in kwargs or "data_parallel_rank" in kwargs:
        raise ValueError(
            "data-parallel coordinates are managed by "
            "generate_data_parallel"
        )

    assignments = [[] for _ in range(data_parallel_size)]
    for index, prompt in enumerate(prompts):
        assignments[index % data_parallel_size].append((index, prompt))

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for replica_id, indexed_prompts in enumerate(assignments):
        process = ctx.Process(
            target=_run_replica,
            args=(
                replica_id,
                data_parallel_size,
                model,
                indexed_prompts,
                sampling_params,
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
