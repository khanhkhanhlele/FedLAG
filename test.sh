cd system
python -u main.py -lbs 16 -nc 20 -jr 1 -data Cifar100 -nb 100 -m cnn20 -algo Recon -gr 6  -jr 1 --top_k 6 -sc 0 -bt 0.001 -go train -fceal --balance --noniid --alpha_dirich 0.1 -did 0
# python main.py -data mnist -m cnn -algo FedAvg -gr 2500 -did 0 -go cnn