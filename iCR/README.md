# iCR: Volumetric Active Mapping
## Installation

```
conda create -n <env-name> python==3.7
conda activate <env-name>
cd .
pip install -r requirements.txt
cd deps/bc-exploration
pip install -e .
conda install pytorch=1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
```