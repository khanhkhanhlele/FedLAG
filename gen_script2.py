import os 
import json
from random import randint 
import random
#import torch
import itertools

if __name__ == "__main__":
    sweep_file = os.getcwd() + "/sweep.json"
    
    sweep_data = json.load(open(sweep_file, mode='r'))
    
    algo = sweep_data["-algo"] 
    ds = sweep_data["-data"]
    nc = sweep_data["-nc"]
    join_ratio = sweep_data["-join_ratio"]
    alpha = sweep_data["-alpha"]
    top_k = sweep_data["--top_k"]
    sc = sweep_data["-sc"]
    
    #gpu_count = torch.cuda.device_count()
    #gpus = range(gpu_count)
    
    script_folder = os.getcwd() + "/scripts_test"
    if not os.path.exists(script_folder):
        os.mkdir(script_folder)
    
    
    for _ds in ds:
        
        for _algo in algo:
            
            #gpu = random.choice(gpus)
            gpu = 0
            
            cmd_lst = ["cd ../system/\n"]
            
            run_file = script_folder + f"/{_ds['name']}_{_algo}.sh"
            
            for _alpha, _nc, _model, _jr, _top_k, _sc in itertools.product(alpha, nc, _ds['-m'], join_ratio, top_k, sc):
                cmd_lst.append(
                    f"python -u main.py -lbs 16 -nc {_nc} -jr 1 -data {_ds['name']} -nb {_ds['#cls']} -m {_model} -algo {_algo} -gr {nc[_nc]}  -jr {_jr} --top_k {_top_k} -sc {_sc} -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich {_alpha} -did {gpu}\n"
                )
            
            
            with open(run_file, mode='w') as file:
                file.writelines(
                    cmd_lst
                )
                file.close()