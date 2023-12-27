cd ../system/
#FedProx+LAG
python -u main.py -lbs 16 -nc 100 -jr 1 -data Cifar10 -nb 10 -m cnn -algo Prox_Rec -gr 800  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 0.1 -did 0

#emnist
python -u main.py -lbs 16 -nc 40 -jr 1 -data emnist -nb 62 -m cnn -algo PerAvg -gr 100  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 40 -jr 1 -data emnist -nb 62 -m cnn -algo FedPAC -gr 100  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 60 -jr 1 -data emnist -nb 62 -m cnn -algo FedPAC -gr 100  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 80 -jr 1 -data emnist -nb 62 -m cnn -algo FedPAC -gr 100  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 80 -jr 1 -data emnist -nb 62 -m cnn -algo FedPAC -gr 100  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 0.1 -did 0
python -u main.py -lbs 16 -nc 100 -jr 1 -data emnist -nb 62 -m cnn -algo FedPAC -gr 100  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 0.1 -did 0
python -u main.py -lbs 16 -nc 100 -jr 1 -data emnist -nb 62 -m cnn -algo FedPAC -gr 100  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 100 -jr 1 -data emnist -nb 62 -m cnn -algo Recon -gr 100  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 100 -jr 1 -data emnist -nb 62 -m cnn -algo FedROD -gr 100  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 100 -jr 1 -data emnist -nb 62 -m cnn -algo FedBABU -gr 100  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 100 -jr 1 -data emnist -nb 62 -m cnn -algo PerAvg -gr 100  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0

#other
python -u main.py -lbs 16 -nc 100 -jr 1 -data Cifar100 -nb 100 -m cnn20 -algo PerAvg -gr 800  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 0.1 -did 0


