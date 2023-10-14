import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_len_map = {
    "20": 100,
    "40" : 200,
    "60" : 400,
    "80" : 600,
    "100" : 800
}

if __name__ == "__main__":
    runs_dir = os.getcwd() + "/system/runs"
    
    runs_files = glob(runs_dir + "/*/*")
    
    ex_dct = {
        "algo" : [],
        "ds" : [],
        "user" : [],
        "train_loss" : [],
        "test_acc" : [],
        "test_auc_std" : [],
        "test_acc_std" : [],
        "test_auc" : []
    }
    
    ds_map = {
        "Cifar100" : "Cifar100",
        "Cifar10" : "Cifar10",
        "mnist" : "MNIST",
        "agnews" : "Agnews"
    }
    
    for run_file in tqdm(runs_files):
        
        algo, ds, user_cnt, _ = run_file.split("/")[-2].split("__")
        
        max_len = run_len_map[user_cnt]
        
        log = EventAccumulator(run_file)
        
        log.Reload()
        
        if len(log.Scalars("charts/test_acc")) < max_len:
            continue
        
        for key in ["train_loss", "test_acc", "test_auc", "test_acc_std", "test_auc_std"]:
            
            ex_dct[key] += [event.value for event in log.Scalars(f"charts/{key}")[:max_len]]
        
        ex_dct['algo'] += [algo]*max_len
        ex_dct['ds'] += [ds_map[ds]]*max_len
        ex_dct['user'] += [user_cnt]*max_len
    
    plot_data_dir = os.getcwd() + "/results_plot"
    if not os.path.exists(plot_data_dir):
        os.mkdir(plot_data_dir)
    
    plot_data_path = plot_data_dir + "/plot_data.csv"
    
    df = pd.DataFrame(ex_dct)
    df.to_csv(plot_data_path)