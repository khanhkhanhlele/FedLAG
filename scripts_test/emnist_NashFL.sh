cd ../system/
python -u main.py -lbs 16 -nc 20 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 100 -bt 0.001 -go train -fceal --log --noniid --alpha_dirich 0.1 -did 0
python -u main.py -lbs 16 -nc 40 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 200 -bt 0.001 -go train -fceal --log --noniid --alpha_dirich 0.1 -did 0
python -u main.py -lbs 16 -nc 60 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 400 -bt 0.001 -go train -fceal --log --noniid --alpha_dirich 0.1 -did 0
python -u main.py -lbs 16 -nc 80 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 600 -bt 0.001 -go train -fceal --log --noniid --alpha_dirich 0.1 -did 0
python -u main.py -lbs 16 -nc 100 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 800 -bt 0.001 -go train -fceal --log --noniid --alpha_dirich 0.1 -did 0
python -u main.py -lbs 16 -nc 20 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 100 -bt 0.001 -go train -fceal --log --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 40 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 200 -bt 0.001 -go train -fceal --log --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 60 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 400 -bt 0.001 -go train -fceal --log --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 80 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 600 -bt 0.001 -go train -fceal --log --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 100 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 800 -bt 0.001 -go train -fceal --log --noniid --alpha_dirich 1 -did 0
python -u main.py -lbs 16 -nc 20 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 100 -bt 0.001 -go train -fceal --log --balance --noniid -did 0
python -u main.py -lbs 16 -nc 40 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 200 -bt 0.001 -go train -fceal --log --balance --noniid -did 0
python -u main.py -lbs 16 -nc 60 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 400 -bt 0.001 -go train -fceal --log --balance --noniid -did 0
python -u main.py -lbs 16 -nc 80 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 600 -bt 0.001 -go train -fceal --log --balance --noniid -did 0
python -u main.py -lbs 16 -nc 100 -jr 1 -data emnist -nb 62 -m cnn -algo NashFL -gr 800 -bt 0.001 -go train -fceal --log --balance --noniid -did 0
