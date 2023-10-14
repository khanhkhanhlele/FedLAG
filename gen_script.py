import os 
import json
from random import randint 
import torch

if __name__ == "__main__":
    sweep_file = os.getcwd() + "/sweep.json"
    
    sweep_data = json.load(open(sweep_file, mode='r'))
    
    algo = sweep_data["-algo"] 
    ds = sweep_data["-data"]
    nc = sweep_data["-nc"]
    go = sweep_data["-go"]
    
    for _ds in ds:
        cmd_lst = []
        run_file = os.getcwd() + f"/{_ds['name']}.sh"
        first_cmd = True
        for _nc in nc:
            if first_cmd:
                prefix = ""
                first_cmd = False
            else:
                prefix = ".."
                
            block_data_cmd = [
                f"cd {prefix}/dataset\n",
                f"python {_ds['gen']} noniid - dir {_nc}\n",
                "cd ../system\n"
            ]
            
            cmd_lst += block_data_cmd
            
            for _algo in algo:
                for _model in 
                cmd_lst.append(
                    f"python -u main.py -lbs 16 -nc {_nc} -jr 1 -data {_ds['name']} -nb {_ds['#cls']} -m {_ds['-m']} -algo {_algo} -gr {nc[_nc]} -did 0 -bt 0.001 -go train\n"
                )
            cmd_lst.append("\n")
        
        with open(run_file, mode='w') as file:
            file.writelines(
                cmd_lst
            )
            file.close()