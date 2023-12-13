import os
import pandas as pd
import numpy as np

alphas = [0.1, 1]
datasets = ["MNIST", "CIFAR10", "CIFAR100", "EMNIST"]
user_counts = [20, 40, 60, 80, 100]
file_path  = "latex_table.txt"


if __name__ == "__main__":
    latex_code = "\section{Results}\n"
    
    for dataset in datasets:
        latex_code += f"\\subsection{{{dataset}}}\n"
        for alpha in alphas:
            latex_code += f"\\subsubsection{{$\\alpha = {alpha}$}}\n"
            latex_code += "\\begin{figure}[!ht]\n"
            latex_code += "    \\centering\n"
            for user_count in user_counts:
                latex_code += f"    \\includegraphics[width=0.195\\linewidth]{{image-lib/benchmark_plot/{dataset.lower()}/{user_count}/test_acc_alpha-{alpha}.pdf}}\n"
                if user_count != user_counts[-1]:
                    latex_code += "    \\hfill\n"
            latex_code += f"    \\caption{{The figures illustrate the performance of FL-LAG vs. various baselines on {dataset}. The evaluation is implemented on different numbers of users (from left to right, the number of users is $U = {','.join(map(str, user_counts))}$, respectively). The sampling rate is set to ${alpha}$}}\n"
            latex_code += f"    \\label{{fig:{dataset.lower()}-alpha-{alpha}}}\n"
            latex_code += "\\end{figure}\n"
        # latex_code += "\\clearpage\n"

    with open(file_path, 'w') as file:
        file.write(latex_code)
