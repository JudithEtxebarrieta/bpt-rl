# To install MuJoCo environments to work with Sample-Factory v2
pip install sample-factory[mujoco]
sudo apt-get install libglew-dev libosmesa6-dev

# PPO with stable_baselines3
pip install stable-baselines3[extra]

# PPO with Gymnasium-Robotics environments
pip install gymnasium-robotics[mujoco-py]

# PPO with Gymnasium-Box2D environmnets
pip install swig
pip install gymnasium[box2d]

# PPO with pybullet environments with gymnasium
pip install pybullet_envs_gymnasium

# virtual environment for brax (https://github.com/google/brax)
python3 -m venv venv/venv
source venv/venv/bin/activate
pip install --upgrade pip
pip install brax
pip install stable-baselines3[extra]

# conda virtual environment for arlbench (https://github.com/automl/arlbench/tree/main)
conda create -n arlbench python=3.10
conda activate arlbench

# virtual environment with the same version of Hipatia virtual environment
sudo apt install python3.9
sudo apt install python3.9-venv
python3.9 -m venv py39venv
source py39venv/bin/activate
pip install stable-baselines3[extra]
pip install gymnasium[mujoco]