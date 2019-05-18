#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 10:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --partition gpgpu
#SBATCH --cpus-per-task=1

module load Python/3.6.4-intel-2017.u2-GCC-6.2.0-CUDA9
module load Tensorflow/1.12.0-intel-2017.u2-GCC-6.2.0-CUDA9-Python-3.6.4-GPU
module load Keras/2.1.5-intel-2017.u2-GCC-6.2.0-CUDA9-Python-3.6.4-GPU
module load Pillow/5.4.1-intel-2017.u2-GCC-6.2.0-CUDA9-Python-3.6.4


python cycle-gan-p2m2.py


