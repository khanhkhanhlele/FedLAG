cd ../system/
python -u main.py -lbs 16 -nc 100 -jr 1 -data Cifar100 -nb 100 -m cnn20 -algo SCAFFOLD_Rec -gr 800  -jr 1 --top_k 2 -sc 0 -bt 0.001 -go train -fceal --log --balance --noniid --alpha_dirich 0.1 -did 0
