import os
import wandb
from glob import glob
import shutil

if __name__ == "__main__":
    api = wandb.Api()
    
    runs = api.runs("scalemind/PFLA")
    
    result_dir = os.getcwd() + "/system/runs"

    run_names = [run.name for run in runs]
    
    redundant_cnt = 0
    
    local_run_dirs = glob(result_dir + "/*")
    
    for local_run_dir in local_run_dirs:
        local_run_name = local_run_dir.split("/")[-1]
        
        if local_run_name not in run_names:
            shutil.rmtree(local_run_dir)
            redundant_cnt += 1
    
    print(f"Number of online runs: {len(run_names)}")
    print(f"Number of local runs: {len(local_run_dirs)}")
    print(f"Number of redundant save dir: {redundant_cnt}")