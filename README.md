# Installation

```
conda create -n <env-name> --file requirements.txt
conda activate <env-name>
conda install pytorch=1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

# Usage
## Tensorboard
```
tensorboard --logdir=/path/to/tensorboard --port <port-num>
```
E.g.
```
tensorboard --logdir=~/RL-for-active-mapping/toy_active_mapping/tensorboard --port 6008
```

## Training
Run
```
python agent.py
```

## Testing
Run
```
python test.py
```
Note: <br>
```env.render(mode='human')``` will display real-time trajectories (i.e. x-y coordinates) of the agent for each episode. An UI is needed for this to work. <br>
```env.render(mode='terminal')``` will only print important variable values in the terminal. <br>

