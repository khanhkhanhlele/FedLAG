import os
import wandb
from glob import glob
import shutil
from tqdm import tqdm
import pandas as pd

run_len_map = {
    "20": 100,
    "40" : 200,
    "60" : 400,
    "80" : 600,
    "100" : 800
}

if __name__ == "__main__":
    api = wandb.Api(timeout=100)
    
    runs = api.runs("scalemind/PFLA")
    
    ex_dct = {
        "algo" : [],
        "ds" : [],
        "user" : [],
        "rounds" : [],
        "test_acc" : [],
        # "train_loss" : [],
        # "test_auc_std" : [],
        # "test_acc_std" : [],
        # "test_auc" : []
        "balance" : [],
        "noniid": [],
        "alpha" : []
    }
    
    ds_map = {
        "Cifar100" : "Cifar100",
        "Cifar10" : "Cifar10",
        "mnist" : "MNIST",
        "emnist" : "EMNIST"
    }
    
    for run in tqdm(runs):
        run_name = run.name
        if run.state not in ["killed", "failed", "crashed"]:            
            if "old" in run.Tags:
                continue
            if run.config["noniid"] != True:
                continue
            
            algo, ds, user_cnt, _ = run_name.split("__")
            
            if ds in ds_map:
                max_len = run_len_map[user_cnt]
                history = run.scan_history()
                
                try:
                    for key in [
                        "test_acc", 
                        # "train_loss", 
                        # "test_auc", 
                        # "test_acc_std", 
                        # "test_auc_std"
                    ]:
                        value_data = [row[f"charts/{key}"] for row in history]
                        if len(value_data) < max_len:
                            raise Exception("max_len")
                        else:
                            clip_data = value_data[:max_len]
                            ex_dct[key] += clip_data
                        
                except Exception as e:
                    print(f"Error: {e} - Run: {run_name}")
                    continue
            
                ex_dct['algo'] += [algo]*max_len
                ex_dct['ds'] += [ds_map[ds]]*max_len
                ex_dct['user'] += [user_cnt]*max_len
                ex_dct['rounds'] += range(max_len)
                
                ex_dct['balance'] += [run.config["balance"]]*max_len
                ex_dct['noniid'] += [run.config["noniid"]]*max_len
                ex_dct['alpha'] += [run.config["alpha_dirich"]]*max_len
            
            else:
                continue
            
        else:
            continue
    
    plot_data_dir = os.getcwd() + "/results_plot"
    if not os.path.exists(plot_data_dir):
        os.mkdir(plot_data_dir)
    
    plot_data_path = plot_data_dir + "/plot_data.csv"
    
    df = pd.DataFrame(ex_dct)
    df.to_csv(plot_data_path)