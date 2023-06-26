import argparse
from pathlib import Path
import subprocess
import yaml

from pretrain import get_args_parser as pretrain_get_args_parser
from eval_linear import get_args_parser as eval_linear_get_args_parser


if __name__ == "__main__":
    slurm_parser = argparse.ArgumentParser("SlurmParser")
    slurm_parser.add_argument("--gpus", default=1, type=int,
                              help="Number of GPUs to use")
    slurm_parser.add_argument("--partition", default="mlhiwidlc_gpu-rtx2080-advanced", type=str,
                              help="The name of the compute partition to use")
    slurm_parser.add_argument("--array", default=0, type=int,
                              help="If n > 0 submits a job array n+1 jobs")
    slurm_parser.add_argument("--time", default="23:59:59", type=str)
    slurm_parser.add_argument("--head", default="simsiam-vanilla", type=str)
    slurm_parser.add_argument("--descr", default="baseline", type=str)
    slurm_parser.add_argument("--exp_dir", default=None, type=str)

    pretrain_parser = pretrain_get_args_parser()
    eval_linear_parser = eval_linear_get_args_parser()

    while True:
        print("Specify slurm parameter: ENTER for default; -h for -help")
        line = input()
        if line == "-h":
            slurm_parser.print_usage()
            continue
        slurm_args = slurm_parser.parse_args(line.split())
        break

    while True:
        print("Specify pretrain parameters: ENTER for default; -h for -help")
        line = input()
        if line == "-h":
            pretrain_parser.print_usage()
            continue
        args = pretrain_parser.parse_args(line.split())
        break

    print("Multiple Seeds? Either type specific seeds or #seeds (0-%seeds) else 0")
    line = input()
    if line:
        if line.startswith('#'):
            num = line.lstrip('#')
            seeds = list(range(int(num)))
        elif len(line.split()):
            seeds = list(map(int, line.split()))
    else:
        seeds = [0]

    while True:
        print("Specify eval parameters: ENTER for default; -h for -help")
        line = input()
        if line == "-h":
            eval_linear_parser.print_usage()
            continue
        eval_linear_args = eval_linear_parser.parse_args(line.split())
        break

    exp_dir = "/work/dlclarge2/rapanti-MinSimAugment/experiments" \
        if slurm_args.exp_dir is None else slurm_args.exp_dir

    if args.data_path is None:
        if args.dataset == "CIFAR10":
            args.data_path = "/work/dlclarge2/rapanti-MinSimAugment/datasets/CIFAR10"
        elif args.dataset == "ImageNet":
            args.data_path = "/data/datasets/ImageNet/imagenet-pytorch"
        else:
            raise ValueError(f"Dataset '{args.dataset}' has no default path. Specify path to dataset.")

    for seed in seeds:
        args.seed = seed
        exp_name = f"{slurm_args.head}-{slurm_args.descr}" \
                   f"-{args.arch}-{args.dataset}-ep{args.epochs}-bs{args.batch_size}" \
                   f"-lr{args.lr}-wd{args.weight_decay}-mom{args.momentum}-seed{args.seed}"
        output_dir = Path(exp_dir).joinpath(exp_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(output_dir)
        print(f"Experiment: {output_dir}")

        # just to make sure that these arguments are the same
        eval_linear_args.arch = args.arch
        eval_linear_args.dataset = args.dataset
        eval_linear_args.data_path = args.data_path
        eval_linear_args.output_dir = args.output_dir

        with open(output_dir.joinpath("pretrain.yaml"), mode="w", encoding="utf-8") as file:
            yaml.safe_dump(dict(vars(args)), file)
        with open(output_dir.joinpath("eval_linear.yaml"), mode="w", encoding="utf-8") as file:
            yaml.safe_dump(dict(vars(eval_linear_args)), file)

        slurm_dir = output_dir.joinpath("slurm")
        slurm_dir.mkdir(parents=True, exist_ok=True)

        code_dir = output_dir.joinpath("code")
        code_dir.mkdir(parents=True, exist_ok=True)
        copy_msg = subprocess.call(["cp", "-r", ".", code_dir])

        slurm_file = slurm_dir.joinpath("%A.%a.%N.txt")
        sbatch = [
            "#!/bin/bash", f"#SBATCH -p {slurm_args.partition}",
            f"#SBATCH -t {slurm_args.time}",
            f"#SBATCH --gres=gpu:{slurm_args.gpus}",
            f"#SBATCH -J {exp_name}",
            f"#SBATCH -o {slurm_file}",
            f"#SBATCH -e {slurm_file}",
            f"#SBATCH --array 0-{slurm_args.array}%1\n" if slurm_args.array > 0 else '',
            'echo "Workingdir: $PWD"',
            'echo "Started at $(date)"',
            'echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"\n',
            "source ~/.profile",
            "conda activate torch"
        ]
        run = [
            "torchrun",
            f"--nproc_per_node={slurm_args.gpus}",
            f"--nnodes=1", f"--standalone",
            f"code/run_pipeline.py"
        ]

        job_file = output_dir.joinpath("job.sh")
        with open(job_file, 'w') as file:
            for line in sbatch:
                file.write(line + " \n")
            file.write("\n")

            for line in run:
                file.write(line + " \\\n")

        out = subprocess.call(["sbatch", job_file], cwd=output_dir)
