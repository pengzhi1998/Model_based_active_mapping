# Landmark-based mapping
Design a RL policy which drives the agent to localize and update the landmarks' positions with fixed steps in a randomized 
environment.
## Installation

```
conda create -n landmark_mapping
conda activate landmark_mapping
pip install -r requirements.txt
cd ./
git clone --branch release_18 https://github.com/Unity-Technologies/ml-agents.git
cd ./ml-agents
python -m pip install mlagents==0.27.0
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
```