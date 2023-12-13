import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set_theme(style="darkgrid")

algo_lst = ["FedLAG", "PerAvg", "FedROD", "FedPAC", "FedBABU", "FedAvg"]
if __name__ == "__main__":
    results_plot_data = os.getcwd() + "/results_plot/plot_data.csv"
    res_df = pd.read_csv(results_plot_data)
    
    bm_plot_dir = results_plot_data = os.getcwd() + "/results_plot/benchmark_plot"
    if not os.path.exists(bm_plot_dir):
        os.mkdir(bm_plot_dir)
    
    ds_lst = res_df['ds'].unique().tolist()
    

    for _ds in tqdm(ds_lst):
        _ds_df = res_df[res_df['ds'] == _ds]
        _ds = _ds.lower()
        ds_plot_dir = bm_plot_dir + f"/{_ds}"
        if not os.path.exists(ds_plot_dir):
            os.mkdir(ds_plot_dir)
        
        ucnt_lst = _ds_df["user"].unique().tolist()
        
        for _unct in ucnt_lst:
            _unct_df = _ds_df[_ds_df['user'] == _unct]
            
            us_plot_dir = ds_plot_dir + f"/{_unct}"
            if not os.path.exists(us_plot_dir):
                os.mkdir(us_plot_dir)
            
            noniid_df = _unct_df[_unct_df["noniid"] == True]
            for _alpha in [0.1, 1, 10]:
                _alpha_df = noniid_df[noniid_df["alpha"] == _alpha]
                if _alpha_df.shape[0] == 0:
                    continue
                data = _alpha_df.groupby(["rounds","algo"]).agg({"test_acc":"max"}).reset_index()
                data = data[data["algo"].isin(algo_lst)]
                
                sns.lineplot(x="rounds", y="test_acc", hue="algo", data=data)
                
                plt.savefig(us_plot_dir + f"/test_acc_alpha-{_alpha}.pdf", dpi=300, format='pdf',bbox_inches='tight')
                
                plt.close()