#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 00:01:00
#SBATCH --gres=gpu:p100:1  
#SBATCH --partition gpgpu
#SBATCH --cpus-per-task=1

module load Python/3.5.2-intel-2017.u2-GCC-5.4.0-CUDA8
module load Tensorflow/1.7.0-intel-2017.u2-GCC-5.4.0-CUDA9-Python-3.5.2-GPU
module load Keras/2.1.5-intel-2017.u2-GCC-5.4.0-CUDA8-Python-3.5.2-GPU
module load Pillow/5.3.0-intel-2016.u3-Python-3.5.2

python cycle-gan-test.py


