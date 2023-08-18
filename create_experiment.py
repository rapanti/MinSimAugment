import argparse
from pathlib import Path
import subprocess
import sys

from utils import find_free_port
from omegaconf import OmegaConf

from pretrain import get_args_parser as pretrain_get_args_parser
from eval_linear import get_args_parser as eval_linear_get_args_parser


if __name__ == "__main__":
    slurm_parser = argparse.ArgumentParser("SlurmParser")
    slurm_parser.add_argument("--gpus", default=1, type=int,
                              help="Number of GPUs to use")
    slurm_parser.add_argument("--partition", default=None, type=str,
                              help="The name of the compute partition to use")
    slurm_parser.add_argument("--array", default=0, type=int,
                              help="If n > 0 submits a job array n+1 jobs")
    slurm_parser.add_argument("--time", default="48:00:00", type=str)
    slurm_parser.add_argument("--head", default="dino", type=str)
    slurm_parser.add_argument("--descr", default=None, type=str)
    slurm_parser.add_argument("--exp_dir", default=None, type=str)

    pretrain_parser = pretrain_get_args_parser()
    eval_linear_parser = eval_linear_get_args_parser()

    conda_env_name = "minsim"
    profile_path = "~/.bashrc"

    if len(sys.argv) > 1:
        slurm_args, rest = slurm_parser.parse_known_args()
        assert slurm_args.descr is not None, "Nasty! Parameter '--descr' empty. Add a meaningful description."

        pretrain_args = pretrain_parser.parse_args(rest)
        path_to_def = f"configs/{pretrain_args.dataset}/pretrain_default.yaml"
        args = OmegaConf.load(path_to_def)
        for arg in vars(pretrain_args):
            value = pretrain_args.__dict__[arg]
            if value is not None:
                args[arg] = value
        seeds = [args.seed]
        path_to_def = f"configs/{pretrain_args.dataset}/eval_linear_default.yaml"
        eval_linear_args = OmegaConf.load(path_to_def)

    else:
        while True:
            print("Specify slurm parameter: ENTER for default; -h for help")
            line = input()
            if line == "-h":
                slurm_parser.print_usage()
                continue
            slurm_args = slurm_parser.parse_args(line.split())
            assert slurm_args.descr is not None, "Nasty! Parameter '--descr' empty. Add a meaningful description."
            break

        while True:
            print("Specify pretrain parameters: ENTER for default; -h for help")
            line = input()
            if line == "-h":
                pretrain_parser.print_usage()
                continue
            temp = pretrain_parser.parse_args(line.split())
            path_to_def = f"configs/{temp.dataset}/pretrain_default.yaml"
            args = OmegaConf.load(path_to_def)
            for arg in vars(temp):
                value = temp.__dict__[arg]
                if value is not None:
                    args[arg] = value
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
            temp = eval_linear_parser.parse_args(line.split())
            path_to_def = f"configs/{args.dataset}/eval_linear_default.yaml"
            eval_linear_args = OmegaConf.load(path_to_def)
            for arg in vars(temp):
                value = temp.__dict__[arg]
                if value is not None:
                    eval_linear_args[arg] = value
            break

    exp_dir = "/gpfs/bwfor/work/ws/fr_ir46-MinSim/experiments" if slurm_args.exp_dir is None else slurm_args.exp_dir

    if args.data_path is None:
        args.data_path = "/gpfs/bwfor/work/ws/fr_ir46-MinSim/data/imagenet"

    if slurm_args.partition is None:
        slurm_args.partition = "single"

    for seed in seeds:
        args.seed = seed
        ps = f"_p{args.patch_size}" if "vit" in args.arch else ""
        ncrops = f"{args.num_global_crops_loader}_{args.num_local_crops_loader}"

        exp_name = f"{slurm_args.head}-{slurm_args.descr}" \
                   f"-{args.arch+ps}-{args.dataset}-ep{args.epochs}-bs{args.batch_size}" \
                   f"-select_{args.select_fn}-ncrops{ncrops}" \
                   f"-lr{args.lr}-wd{args.weight_decay}-out_dim{str(args.out_dim//1000)+'k'}-seed{args.seed}"
        output_dir = Path(exp_dir).joinpath(exp_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define master port (for preventing 'Address already in use error' when submitting more than 1 jobs on 1 node)
        master_port = find_free_port()
        args.dist_url = "tcp://localhost:" + str(master_port)

        # make sure that these arguments are the same
        eval_linear_args.arch = args.arch
        eval_linear_args.dataset = args.dataset
        eval_linear_args.data_path = args.data_path
        eval_linear_args.dist_url = args.dist_url
        eval_linear_args.dist_backend = args.dist_backend
        eval_linear_args.num_workers = args.num_workers
        eval_linear_args.output_dir = args.output_dir

        print(f"using {args.dist_url=}")
        print(f"Experiment: {output_dir}")

        with open(output_dir.joinpath("pretrain.yaml"), mode="w", encoding="utf-8") as file:
            OmegaConf.save(config=args, f=file)
        with open(output_dir.joinpath("eval_linear.yaml"), mode="w", encoding="utf-8") as file:
            OmegaConf.save(config=eval_linear_args, f=file)

        slurm_dir = output_dir.joinpath("slurm")
        slurm_dir.mkdir(parents=True, exist_ok=True)

        code_dir = output_dir.joinpath("code")
        code_dir.mkdir(parents=True, exist_ok=True)
        copy_msg = subprocess.call(["cp", "-r", ".", code_dir])

        slurm_file = slurm_dir.joinpath("%A.%a.%N.txt")
        ntasks = slurm_args.gpus * 8
        mem = slurm_args.gpus * 32
        sbatch = [
            "#!/bin/bash",
            f"#SBATCH -p {slurm_args.partition}",
            f"#SBATCH -t {slurm_args.time}",
            f"#SBATCH --ntasks-per-node={ntasks}",
            f"#SBATCH --gres=gpu:{slurm_args.gpus}",
            f"#SBATCH --mem={mem}G",
            f"#SBATCH -J {exp_name}",
            f"#SBATCH -o {slurm_file}",
            f"#SBATCH -e {slurm_file}",
            f"#SBATCH --array 0-{slurm_args.array}%1\n" if slurm_args.array > 0 else '',
            'echo "Workingdir: $PWD"',
            'echo "Started at $(date)"',
            'echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"\n',
            "module load devel/cuda",
            "module load 'devel/miniconda/3'",
            f"source {profile_path}",
            f"conda activate {conda_env_name}"
        ]
        run = [
            "torchrun",
            f"--nproc_per_node={slurm_args.gpus}",
            f"--nnodes=1",
            # rdzv assigns ports automatically to workers when port=0, however DDP uses its own 'master_port'
            f"--rdzv-endpoint=localhost:0",
            f"--rdzv-backend=c10d",
            f"--rdzv-id=$SLURM_JOB_ID",
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
