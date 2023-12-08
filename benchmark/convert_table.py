import os
import pandas as pd
import numpy as np

columns = [
    'algo', 
    'mnist_0.1', 'cifar10_0.1', 'cifar100_0.1', 'emnist_0.1',
    'mnist_1', 'cifar10_1', 'cifar100_1', 'emnist_1',
    'mnist_10', 'cifar10_10', 'cifar100_10', 'emnist_10',
    'user'
]
if __name__ == "__main__":
    results_plot_data = os.getcwd() + "/results_plot/benchmark.csv"
    df = pd.read_csv(results_plot_data)
    table = np.empty((0,14))
    for user in [20, 40, 60, 80, 100]:
        
        df_1 = df[df["user"] == user]
        for algo in ["Recon", "PerAvg","FedROD", "FedPAC", "FedBABU", "FedAvg"]:
            
            df_2 = df_1[df_1["algo"] == algo]
            arr= np.array([algo])
            for alpha in [0.1, 1, 10]:
                
                df_3 = df_2[df_2["alpha"] == alpha]
                for ds in ["MNIST", "Cifar10", "Cifar100", "EMNIST"]:
                    
                    df_4 =df_3[df_3["ds"] == ds]["test_acc"]
                    data = df_4.max() if not df_4.empty else 0
                    data =  round(data*100, 2)
                    arr = np.append(arr, data)
            arr = np.append(arr, user)
            table = np.append(table, [arr], axis=0)

    file_path = os.getcwd() + "/results_plot/table.csv"
    save_table = pd.DataFrame(table)
    save_table.to_csv(file_path, index=False)
        
    
    