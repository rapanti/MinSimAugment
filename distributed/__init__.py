import os
import sys

import torch
import torch.distributed as dist


def ddp_setup() -> (int, int, int):
    # launched with torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        _rank = int(os.environ["RANK"])
        _world_size = int(os.environ['WORLD_SIZE'])
        _local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            rank=_local_rank,
            world_size=_world_size,
        )

        torch.cuda.set_device(_local_rank)
        print('Initialized distributed on rank: {}'.format(_rank), flush=True)
        dist.barrier()
        _restrict_print_to_main_process()
        return _local_rank, _rank, _world_size

    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        _rank, _local_rank, _world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            rank=_local_rank,
            world_size=_world_size,
        )
        torch.cuda.set_device(_local_rank)
        return _local_rank, _rank, _world_size

    else:
        print('Does not support training without GPU.')
        sys.exit(1)


def is_enabled() -> bool:
    """True if distributed training is enabled."""
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    """Returns the current world size (number of distributed processes)."""
    if not is_enabled():
        return 1
    return dist.get_world_size()


def rank() -> int:
    """Returns the rank of the current process within the global process group."""
    return dist.get_rank() if is_enabled() else 0


def is_main_process() -> bool:
    """Returns True if the current process is the main one."""
    return rank() == 0


def rank_zero_only(fn):
    """Decorator that only runs the function on the process with rank 0."""
    def wrapped(*args, **kwargs):
        if rank() == 0:
            return fn(*args, **kwargs)

    return wrapped


@rank_zero_only
def print_rank_zero(*args, **kwargs) -> None:
    """Equivalent to print, but only runs on the process with rank 0."""
    print(*args, **kwargs)


def _restrict_print_to_main_process() -> None:
    """
    This function disables printing when not in the main process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):  # noqa
        force = kwargs.pop("force", False)
        if is_main_process() or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def _is_slurm_job_process() -> bool:
    return "SLURM_JOB_ID" in os.environ
