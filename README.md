# Landmark-based mapping
Design a RL policy which drives the agent to localize and update the landmarks' positions with fixed steps in a randomized 
environment.
The yaml files are borrowed from this great repo: https://github.com/ehfd/docker-nvidia-glx-desktop.git and
https://ucsd-prp.gitlab.io/userdocs/running/gui-desktop/
## Installation

```
conda create -n landmark_mapping python==3.8 -y
conda activate landmark_mapping
pip install -r requirements.txt
cd ./
git clone --branch release_18 https://github.com/Unity-Technologies/ml-agents.git
cd ./ml-agents
pip install -e ./ml-agents-envs
pip install gym-unity==0.27.0
```
