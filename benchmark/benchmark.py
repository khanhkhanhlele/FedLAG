import os
import pandas as pd
import numpy as np

if __name__ == "__main__":
    results_plot_data = os.getcwd() + "/results_plot/plot_data.csv"
    res_df = pd.read_csv(results_plot_data)
    
    res_gr_df = res_df.groupby(
        by = ["algo", "ds", "user"]
    ).agg(
        {
            'train_loss': 'min', 
            'test_acc': 'max',
            'test_auc_std' : 'min',
            'test_acc_std' : 'min',
            'test_auc' : 'max'
        }
    )
    
    benchmark_file = os.getcwd() + "/results_plot/benchmark.csv"
    
    res_gr_df.to_csv(benchmark_file)
    
    res_gr_df = pd.read_csv(benchmark_file)
    
    for ds_key, ds_name in zip(
        ['Cifar10', 'Cifar100', 'MNIST', 'Agnews'],
        ['cifar10', 'cifar100', 'mnist', 'agnews']
    ):

        res_gr_df_ds = res_gr_df[res_gr_df['ds'] == ds_key]
        if res_gr_df_ds.shape[0] == 0:
            continue
        res_gr_df_ds.to_csv(os.getcwd() + f"/results_plot/benchmark_{ds_name}.csv")