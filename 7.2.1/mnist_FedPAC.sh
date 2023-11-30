cd ../system/
python -u main.py -lbs 16 -nc 100 -jr 1 -data mnist -nb 10 -m dnn -algo FedPAC -gr 800  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --noniid --alpha_dirich 0.1 -did 1
python -u main.py -lbs 16 -nc 100 -jr 1 -data mnist -nb 10 -m dnn -algo FedPAC -gr 800 -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance -did 1
