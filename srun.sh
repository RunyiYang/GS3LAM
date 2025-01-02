#!/bin/bash
#SBATCH --job-name=GS3LAM
#SBATCH --partition=batch
#SBATCH --gpus=l4-24g:1          # Specify the partition
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --cpus-per-task=8       # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=8G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=2-0         # Job timeout
#SBATCH --output=output_logs/test.log      # Redirect stdout to a log file
#SBATCH --error=output_logs/test.error     # Redirect stderr to a separate error log file
#SBATCH --mail-type=ALL         # Send updates via email


micromamba create -n gs3lam python==3.10 -y
export CUDA_HOME=/opt/modules/nvidia-cuda-11.8.0
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH
export PATH=/opt/modules/gcc-11.5.0/bin:$PATH
export CC=/opt/modules/gcc-11.5.0/bin/gcc-11.5
export CXX=/opt/modules/gcc-11.5.0/bin/g++-11.5
export LD=/opt/modules/gcc-11.5.0/bin/g++-11.5
export TORCH_CUDA_ARCH_LIST="6.0+PTX"

eval "$(micromamba shell hook --shell bash)"
micromamba activate
micromamba activate gs3lam
micromamba install -c "nvidia/label/cuda-11.7.0" cuda-toolkit -y

nvcc -V
nvidia-smi

pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# install Gaussian Rasterization
pip install submodules/gaussian-semantic-rasterization

srun python run.py configs/Replica/replica.py