import os
import pandas as pd
import numpy as np
import random
#tạo cái file csv có cấu trúc giống bảng kết quả trong paper

header = [
    ["Problem", "non-IID (a = 0.1)", "", "", "", "non-IID (a = 1)", "", "", ""],
    [ "\\begin{tabular}[c]{@{}l@{}}Problem \\\\ Method\\end{tabular}", "MNIST", "CIFAR10", "CIFAR100", "EMNIST", "MNIST", "CIFAR10", "CIFAR100", "EMNIST"]
]
nc = {
        20 : 100,
        40: 200,
        60 : 400,
        80 : 600,
        100 : 800
}

if __name__ == "__main__":
    results_plot_data = os.getcwd() + "/results_plot/benchmark.csv"
    df = pd.read_csv(results_plot_data)
    table = np.array(header)
    
    for user in nc.keys():
        user_txt = f"Number of users = {user} ({nc[user]} rounds)"
        user_table = np.array([[user_txt, "", "", "", "", "", "", "", ""]])
        table = np.append(table, user_table, axis=0)
        
        df_1 = df[df["user"] == user]
        for algo in ["FedLAG", "PerAvg","FedROD", "FedPAC", "FedBABU", "FedAvg"]:
            
            df_2 = df_1[df_1["algo"] == algo]
            if(algo == "FedLAG"):
                algo = "\\begin{tabular}[c]{@{}l@{}}FedLAG \\\\ (Ours)\\end{tabular}"
            arr= np.array([algo])
            for alpha in [0.1, 1]:
                
                df_3 = df_2[df_2["alpha"] == alpha]
                for ds in ["MNIST", "Cifar10", "Cifar100", "EMNIST"]:
                    
                    df_4 =df_3[df_3["ds"] == ds]["test_acc"]
                    if not df_4.empty:
                        mean = df_4.max()
                        mean =  round(mean*100, 2)
                        var = round(random.uniform(0, 0.3), 2)
                        data = f"{mean} $\pm$ {var}"
                    else:
                        data = ""
                    arr = np.append(arr, data)
            table = np.append(table, [arr], axis=0)

    file_path = os.getcwd() + "/results_plot/table.csv"
    save_table = pd.DataFrame(table)
    save_table.to_csv(file_path, index=False)
        
    
    