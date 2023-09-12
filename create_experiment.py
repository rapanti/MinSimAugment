import argparse
from pathlib import Path
import subprocess
import getpass

from omegaconf import OmegaConf

from pretrain import get_pretrain_args_parser
from linear import get_linear_args_parser
from finetune import get_finetune_args_parser
from knn import get_knn_args_parser

DEF_KNN = "configs/knn_default.yaml"
DEF_LINEAR = "configs/linear_default.yaml"


if __name__ == "__main__":
    slurm_parser = argparse.ArgumentParser("SlurmParser")
    slurm_parser.add_argument("--gpus", default=1, type=int,
                              help="Number of GPUs to use")
    slurm_parser.add_argument("-p", "--partition", default="alldlc_gpu-rtx2080", type=str,
                              help="The name of the compute partition to use")
    slurm_parser.add_argument("-a", "--array", default=0, type=int,
                              help="If n > 0 submits a job array n+1 jobs")
    slurm_parser.add_argument("-t", "--time", default="1-00:00:00", type=str)
    slurm_parser.add_argument("-j", "--job-name", default=None, type=str)
    slurm_parser.add_argument("--head", default="simclr_v1", type=str)
    slurm_parser.add_argument("-d", "--descr", default=None, type=str)
    slurm_parser.add_argument("--seed", default=None, type=int)
    slurm_parser.add_argument("--exp_dir", default=None, type=str)
    slurm_parser.add_argument("--manual", action="store_true")

    pretrain_parser = get_pretrain_args_parser()
    eval_knn_parser = get_knn_args_parser()
    eval_linear_parser = get_linear_args_parser()

    slurm_args, rest = slurm_parser.parse_known_args()
    assert (slurm_args.descr is not None) or (slurm_args.job_name is not None),\
        "Nasty! Parameter '--descr' (or --job-name) empty. Add a meaningful description."

    current_username = getpass.getuser()
    conda_env_name = "torch" if current_username == "rapanti" else "minsim2"
    profile_path = "~/.profile" if current_username == "rapanti" else "/home/ferreira/.profile"

    args = pretrain_parser.parse_args(rest)
    args = OmegaConf.create(vars(args))

    eval_args_knn = OmegaConf.load(DEF_KNN)
    eval_args_linear = OmegaConf.load(DEF_LINEAR)

    if slurm_args.exp_dir is None:
        if current_username == "rapanti":
            exp_dir = "/work/dlclarge2/rapanti-MinSimAugment/experiments"
        else:
            exp_dir = "/work/dlclarge1/ferreira-simsiam/minsim_experiments"
    else:
        exp_dir = slurm_args.exp_dir

    if args.data_path is None:
        if args.dataset.lower() == "cifar10":
            if current_username == "rapanti":
                args.data_path = "/work/dlclarge2/rapanti-MinSimAugment/datasets/CIFAR10"
            else:
                args.data_path = "/work/dlclarge1/ferreira-simsiam/simsiam/datasets/CIFAR10"
        elif args.dataset.lower() == "imagenet":
            args.data_path = "/data/datasets/ImageNet/imagenet-pytorch"
        else:
            raise ValueError(f"Dataset '{args.dataset}' has no default path. Specify path to dataset.")

    if slurm_args.partition is None:
        slurm_args.partition = "alldlc_gpu-rtx2080"

    ps = f"_p{args.patch_size}" if "vit" in args.arch else ""
    exp_name = f"{slurm_args.head}-{slurm_args.descr}" \
               f"-{args.arch+ps}-{args.dataset}-ep{args.epochs}-bs{args.batch_size}" \
               f"-lr{args.lr}-seed{args.seed}"
    if slurm_args.job_name is not None:
        exp_name = slurm_args.job_name
    output_dir = Path(exp_dir).joinpath(exp_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # make sure that these arguments are the same
    eval_args_knn.arch = eval_args_linear.arch = args.arch
    eval_args_knn.arch_kwargs = eval_args_linear.arch_kwargs = args.arch_kwargs
    eval_args_knn.dataset = eval_args_linear.dataset = args.dataset
    eval_args_knn.data_path = eval_args_linear.data_path = args.data_path
    eval_args_knn.output_dir = eval_args_linear.output_dir = args.output_dir

    print(f"Experiment: {output_dir}")

    with open(output_dir.joinpath("pretrain.yaml"), mode="w", encoding="utf-8") as file:
        OmegaConf.save(config=args, f=file)

    with open(output_dir.joinpath("linear.yaml"), mode="w", encoding="utf-8") as file:
        OmegaConf.save(config=eval_args_linear, f=file)

    with open(output_dir.joinpath("knn.yaml"), mode="w", encoding="utf-8") as file:
        OmegaConf.save(config=eval_args_knn, f=file)

    slurm_dir = output_dir.joinpath("slurm")
    slurm_dir.mkdir(parents=True, exist_ok=True)

    code_dir = output_dir.joinpath("code")
    code_dir.mkdir(parents=True, exist_ok=True)
    copy_msg = subprocess.call(["cp", "-r", ".", code_dir])

    slurm_file = slurm_dir.joinpath("%A.%a.%N.txt")
    sbatch = [
        "#!/bin/bash",
        f"#SBATCH -p {slurm_args.partition}",
        f"#SBATCH -t {slurm_args.time}",
        f"#SBATCH --gres=gpu:{slurm_args.gpus}",
        f"#SBATCH -J {exp_name}",
        f"#SBATCH -o slurm/%A.%a.%N.txt",
        f"#SBATCH -a 0-{slurm_args.array}%1\n" if slurm_args.array > 0 else '',
        'echo "Workingdir: $PWD"',
        'echo "Started at $(date)"',
        'echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"\n',
        f"source {profile_path}",
        f"conda activate {conda_env_name}"
    ]
    run = [
        "torchrun",
        f"--nnodes=1",
        f"--nproc_per_node={slurm_args.gpus}",
        f"--rdzv-endpoint=localhost:0",
        f"--rdzv-backend=c10d",
        f"--rdzv-id=$SLURM_JOB_ID",
        f"--max-restart=3",
        f"code/main.py"
    ]

    job_file = output_dir.joinpath("job.sh")
    with open(job_file, 'w') as file:
        for line in sbatch:
            file.write(line + "\n")
        file.write("\n")

        for line in run:
            file.write(line + " \\\n")

    if not slurm_args.manual:
        out = subprocess.call(["sbatch", job_file], cwd=output_dir)
