#!/bin/bash

#SBATCH --job-name=diffuseq
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=5g
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --account=eecs598s007f22_class
#SBATCH --partition=spgpu
#SBATCH --mail-user=yiweilyu@umich.edu
#SBATCH --export=ALL

python -m torch.distributed.run run_decode.py \
--model_dir /mnt/c/TopicsInDL/diffusion_models/diffuseq_PPR_h128_lr0.0001_t1000_linear_lossaware_seed103_PPR20241127-13:49:54 \
--pattern ema_0.9999_005500





