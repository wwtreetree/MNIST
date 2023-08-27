import os
import socket
from subprocess import run, PIPE
from time import sleep
import itertools as it


def set_env(command, cluster):
    os.environ['COMMAND'] = command
    os.environ['WANDB_ENTITY'] = "wtree"

    if cluster == 'submit':
        os.environ['VENV'] = "/ubc/cs/home/w/weilbach/Development/algorithm_from_conditioning/gsdm_env"
    elif cluster == 'cedar':
        os.environ['VENV'] = "/home/shuwang/scratch/jupyternote/mnist_env"
    else:
        os.environ['VENV'] = "/ubc/cs/home/w/weilbach/Development/algorithm_from_conditioning/gsdm_env"
    output_dir = os.path.join(os.environ['SCRATCH'], 'slurm-outputs')


def submit(command, jobname, machine=None):
    host = socket.gethostname()
    cluster = "submit" if (host == 'submit-ml') else "cedar" if "cedar" in host else "plai"
    partition = 'ubcml-rti' if cluster=='submit' else "def-schmidtm" if cluster=='cedar' else 'plai'
    set_env(command, cluster)
    output_dir = "/home/shuwang/scratch/slurm-outputs/" #os.path.join(os.environ['SCRATCH'], 'slurm-outputs')
    sbatch = f"sbatch{' -w '+machine if machine is not None else ''} --export=all --account {partition} -t 3:0:0 --mem=10G --gres=gpu:1 -J {jobname} --output=\"{output_dir}/mnist-%j.out\" --error=\"{output_dir}/mnist-%j.err\" job.sh"
    result = run(sbatch, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True).stdout
    msg = sbatch + '\n' + str(command) + '\n' + str(result) + '\n'
    print(msg)
    with open('jobs.txt', 'a') as f:
        f.write(msg)
    sleep(1)

base = "/home/shuwang/scratch/jupyternote"
for lr in [0.01, 0.02, 0.05, 0.1]:
    for epoch in [4,5,6]:
        for activation in ["sigmoid","tanh", "relu"]:
            for optimizer in ["SGD", "Adam"]:
                command = f"python3 {base}/MNIST_wandb.py --lr={lr} --epoch={epoch} --activation={activation} --optimizer={optimizer}"
                submit(command, jobname=f"lr_{lr}, epoch_{epoch}, activation_{activation}, optimizer_{optimizer}")

