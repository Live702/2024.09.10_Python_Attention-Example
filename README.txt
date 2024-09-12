### Steps used to create env_1:
- Run ps cmd:  ```conda create --name env_1 python=3.9```
- Run ps cmd: ```conda activate env_1```
- Run ps cmd: ```conda install pytorch torchvision torchaudio cpuonly -c pytorch```
- Run ps cmd: ```conda install pandas matplotlib scikit-learn```
- Run ps cmd: ```conda install tensorflow```
- Run ps cmd: ```conda install matplotlib```
- Run ps cmd: ```conda install -c conda-forge tensorboard```
- Run ps cmd: ```conda install -c conda-forge tqdm```

### Steps to import env_1:
- Run ps cmd: ```conda env create -f env_1.yml```
- Run ps cmd: ```conda install pytorch torchvision torchaudio cpuonly -c pytorch```

### User guide:
- Ensure conda is installed and configured (see my other example repo)
- Import or re-create env_1
- Open another powershell window and cd to the repo dir
- Run ps cmd: ```tensorboard --logdir=runs```
