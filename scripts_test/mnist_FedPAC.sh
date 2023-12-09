cd ../system/
python -u main.py -lbs 16 -nc 20 -jr 1 -data mnist -nb 10 -m dnn -algo FedPAC -gr 100  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 40 -jr 1 -data mnist -nb 10 -m dnn -algo FedPAC -gr 200  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 60 -jr 1 -data mnist -nb 10 -m dnn -algo FedPAC -gr 400  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 80 -jr 1 -data mnist -nb 10 -m dnn -algo FedPAC -gr 600  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 100 -jr 1 -data mnist -nb 10 -m dnn -algo FedPAC -gr 800  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 20 -jr 1 -data mnist -nb 10 -m dnn -algo FedPAC -gr 100  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 10 -did 0
python -u main.py -lbs 16 -nc 40 -jr 1 -data mnist -nb 10 -m dnn -algo FedPAC -gr 200  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 10 -did 0
python -u main.py -lbs 16 -nc 60 -jr 1 -data mnist -nb 10 -m dnn -algo FedPAC -gr 400  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 10 -did 0
python -u main.py -lbs 16 -nc 80 -jr 1 -data mnist -nb 10 -m dnn -algo FedPAC -gr 600  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 10 -did 0
python -u main.py -lbs 16 -nc 100 -jr 1 -data mnist -nb 10 -m dnn -algo FedPAC -gr 800  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 10 -did 0
