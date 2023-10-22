import os 
import json
from random import randint 
import random
import torch

if __name__ == "__main__":
    sweep_file = os.getcwd() + "/sweep.json"
    
    sweep_data = json.load(open(sweep_file, mode='r'))
    
    algo = sweep_data["-algo"] 
    ds = sweep_data["-data"]
    nc = sweep_data["-nc"]
    
    gpu_count = torch.cuda.device_count()
    gpus = range(gpu_count)
    
    script_folder = os.getcwd() + "/scripts"
    if not os.path.exists(script_folder):
        os.mkdir(script_folder)
    
    
    for _ds in ds:
        
        for _algo in algo:
            
            gpu = random.choice(gpus)
            
            cmd_lst = ["cd ../system/\n"]
            
            run_file = script_folder + f"/{_ds['name']}_{_algo}.sh"
            
            for _nc in nc:
                for _model in _ds['-m']:
                    cmd_lst.append(
                        f"python -u main.py -lbs 16 -nc {_nc} -jr 1 -data {_ds['name']} -nb {_ds['#cls']} -m {_model} -algo {_algo} -gr {nc[_nc]} -did 0 -bt 0.001 -go train -fceal --log --noniid -did {gpu}\n"
                    )
            
            with open(run_file, mode='w') as file:
                file.writelines(
                    cmd_lst
                )
                file.close()