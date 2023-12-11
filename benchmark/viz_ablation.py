import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import product
sns.set_theme(style="darkgrid")

# phần này dùng để vẽ biểu đồ so sánh các phương pháp + LAG trong mục ablation

if __name__ == "__main__":
    results_plot_data = os.getcwd() + "/results_plot/plot_data.csv"
    res_df = pd.read_csv(results_plot_data)
    
    bm_plot_dir = results_plot_data = os.getcwd() + "/results_plot/viz_ablation"
    if not os.path.exists(bm_plot_dir):
        os.mkdir(bm_plot_dir)
    
    algo_lst = [["Ditto", "Ditto_Rec"],
                ["FedProx", "Prox_Rec"],
                ["FedBABU", "Babu_Rec"],
                ["SCAFFOLD", "SCAFFOLD_Rec"],
                ["FedROD", "Rod_Rec"]]
    
    # Define colors for each pair
    colors = sns.color_palette("tab10", len(algo_lst))
    
    res_df = res_df[res_df["algo"].isin([algo for sublist in algo_lst for algo in sublist])]
    
    script = {
        "ds": ["Cifar10"],
        "user": [100],
        "alpha": [0.1]
        }
        
    for _ds, _user, _alpha in product(*script.values()):
        res_df_tmp = res_df[(res_df["ds"] == _ds) & (res_df["user"] == _user) & (res_df["alpha"] == _alpha)]
        
        ds_plot_dir = bm_plot_dir + f"/{_ds}_{_user}_{_alpha}.pdf"

        
        for idx, (algo1, algo2) in enumerate(algo_lst):
            # change name
            if "_Rec" in algo2:
                algo2_label = algo1 + "+LAG"
            sns.lineplot(x="rounds", y="test_acc", data=res_df_tmp[res_df_tmp['algo'] == algo1], label=algo1, color=colors[idx], linestyle='-')
            sns.lineplot(x="rounds", y="test_acc", data=res_df_tmp[res_df_tmp['algo'] == algo2], label=algo2_label, color=colors[idx], linestyle='--')

        plt.savefig(ds_plot_dir, dpi=300, format='pdf')
        plt.close()    
                
          

                