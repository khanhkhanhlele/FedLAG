cd system
python -u main.py -lbs 16 -nc 20 -jr 1 -data mnist -nb 10 -m cnn -algo Recon -gr 3  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --balance --noniid --alpha_dirich 0.1 -did 0
# python main.py -data mnist -m cnn -algo FedAvg -gr 2500 -did 0 -go cnn