cd ../system/
python -u main.py -lbs 16 -nc 100 -jr 1 -data emnist -nb 62 -m cnn -algo Recon -gr 800  -jr 0.2 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 0.1 -did 1
python -u main.py -lbs 16 -nc 100 -jr 1 -data emnist -nb 62 -m cnn -algo Recon -gr 800  -jr 0.5 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 0.1 -did 1
python -u main.py -lbs 16 -nc 100 -jr 1 -data emnist -nb 62 -m cnn -algo Recon -gr 800  -jr 1.0 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 0.1 -did 1
