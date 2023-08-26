import os
import sys

import torch
import torch.distributed as dist


def init_distributed_mode(
        dist_backend: str = "nccl",
        dist_url: str = "env://",
):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    # elif 'SLURM_PROCID' in os.environ:
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        rank, gpu, world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend=dist_backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(gpu)
    print('| distributed init (rank {}): {}'.format(
        rank, dist_url), flush=True)
    dist.barrier()
    _restrict_print_to_main_process()
    return gpu, rank, world_size


def is_enabled() -> bool:
    """
    Returns:
        True if distributed training is enabled
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not is_enabled():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """
    Returns:
        The rank of the current process within the global process group.
    """
    return dist.get_rank() if is_enabled() else 0


def is_main_process() -> bool:
    """
    Returns:
        True if the current process is the main one.
    """
    return get_rank() == 0


def _restrict_print_to_main_process() -> None:
    """
    This function disables printing when not in the main process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main_process() or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def _is_slurm_job_process() -> bool:
    return "SLURM_JOB_ID" in os.environ
