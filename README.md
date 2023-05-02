# Landmark-based mapping
This repository is a PyTorch implementation for paper ***[Policy Learning for Active Target Tracking over Continuous SE(3)
Trajectories](https://arxiv.org/pdf/2212.01498.pdf)***
in L4DC 2023. Authors: [Pengzhi Yang](https://pengzhi1998.github.io/), [Shumon Koga](https://shumon0423.github.io/), [Arash Asgharivaskasi](https://arashasgharivaskasi-bc.github.io/), 
[Nikolay Atanasov](https://natanaso.github.io/).
If you are using the code for research work, please cite:
```
@inproceedings{yang2023l4dc,
  title={Policy Learning for Active Target Tracking over Continuous SE(3) Trajectories},
  author={Yang, Pengzhi and Koga, Shumon and Asgharivaskasi, Arash and Atanasov, Nikolay},
  booktitle={Learning for Dynamics and Control (L4DC)},
  year={2023}
}
```

[//]: # (Design a RL policy which drives the agent to localize and update the landmarks' positions with fixed steps in a randomized )

[//]: # (environment.)

[//]: # (The yaml files are borrowed from this great repo: https://github.com/ehfd/docker-nvidia-glx-desktop.git and)

[//]: # (https://ucsd-prp.gitlab.io/userdocs/running/gui-desktop/)
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
