# Official Implementation of Which layers should undergo personalization in Federated Learning?

![alt text](https://github.com/khanhkhanhlele/FL_recon/blob/main/imgs/DGA.png)

# Envinronment Setup
```
python3 -m venv .env
source .env/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```
The project is conducted using the Nvidia Driver version of ```535.xx``` and CUDA version of ```12.1```, hence Pytorch can be installed as follows:
```
pip3 install torch torchvision torchaudio
```

# Data Setup
To generate data set in various configuration first move your pointer into ```/path/to/FedLAG/dataset```
```
cd /path/to/FedLAG/dataset
```
## Generate IDD & Non IID scenario
To generate dataset with IDD or Non IDD scenario, user need to adjust the value fed into the first arg of each generator file (i.e. generate_cifar10.py). 
```
python generate_cifar10.py noniid # noniid case
python generate_cifar10.py iid # iid case
```
To set balance or imbalance distribution
```
python generate_cifar10.py noniid balance # balance case
python generate_cifar10.py noniid - # long-tailed distribution case
```

## Generate Data set for different number of users along with different distribution 

Use the following command to generate data set with different number (i.e. 20, 40, 60, 80, 100) of users and ```alpha``` factor (i.e. 0.1, 1) of Dirichlet distribution
```
python generate_cifar10.py iid - dir 20 0.1
python generate_cifar10.py iid - dir 40 0.1 
python generate_cifar10.py iid - dir 60 0.1 
python generate_cifar10.py iid - dir 80 0.1
python generate_cifar10.py iid - dir 100 0.1
python generate_cifar10.py noniid - dir 20 0.1
python generate_cifar10.py noniid - dir 40 0.1
python generate_cifar10.py noniid - dir 60 0.1
python generate_cifar10.py noniid - dir 80 0.1
python generate_cifar10.py noniid - dir 100 0.1
```
the mid param is used to bump data set into partition. 

# Simulation Conducting
After generate needed data set, repo user can conduct experiment. First cd to ```system``` folder.
```
cd /path/to/FedLAG/system
``` 
then use the params in ```main.py``` to conduct the simulation, the follow command is an example:
```
python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m dnn -algo FedFomo -gr 2000 -M 5 -did 1 -go dnn --log
```

## Wandb and Tensorboard
If you want to track your experiment, consider use ```--log``` args, toggle it to use the wandb and tensorboard along with a automated folder structure.

## Benchmark and Visualization
If you want to clone all experiments from Wandb and visualize them, consider use the two following files:
- ```/path/to/FedLAG/benchmark/gather_online_data.py``` to gather all experiments
- ```/path/to/FedLAG/benchmark/benchmark.py``` to create an offline benchmark, which is saved to ```/path/to/FedLAG/benchmark/results_plot/```

# Citation
```
```
