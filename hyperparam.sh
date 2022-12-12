#!usr/bin/bash

python experiments.py --img_lambda 0.33 --txt_lambda 0.33 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.33 --txt_lambda 0.33 --hidden_dim 256 --learning_rate 1e-4
python experiments.py --img_lambda 0.33 --txt_lambda 0.33 --hidden_dim 512 --learning_rate 1e-4
python experiments.py --img_lambda 0.33 --txt_lambda 0.33 --hidden_dim 128 --learning_rate 1e-3
python experiments.py --img_lambda 0.33 --txt_lambda 0.33 --hidden_dim 128 --learning_rate 5e-4