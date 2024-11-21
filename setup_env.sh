conda create -y -n rl_environment python=3.10
source ~/.bashrc
conda activate rl_environment
conda install -y -c menpo osmesa
mkdir -p $HOME/.mujoco
curl -L -o $HOME/.mujoco/download.tar.gz https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
cd $HOME/.mujoco/
tar -xvzf download.tar.gz 
cd -
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
sudo yum install -y patchelf
sudo yum install -y libX11-devel
sudo yum install -y glew-devel
source ~/.bashrc
conda activate rl_environment
pip install -r requirements.txt