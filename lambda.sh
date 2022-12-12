#!usr/bin/bash

python experiments.py --img_lambda 0 --txt_lambda 0 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0 --txt_lambda 0.2 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0 --txt_lambda 0.4 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0 --txt_lambda 0.6 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0 --txt_lambda 0.8 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0 --txt_lambda 1 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.2 --txt_lambda 0 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.2 --txt_lambda 0.2 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.2 --txt_lambda 0.4 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.2 --txt_lambda 0.6 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.2 --txt_lambda 0.8 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.4 --txt_lambda 0 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.4 --txt_lambda 0.2 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.4 --txt_lambda 0.4 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.4 --txt_lambda 0.6 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.6 --txt_lambda 0 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.6 --txt_lambda 0.2 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.6 --txt_lambda 0.4 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.8 --txt_lambda 0 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 0.8 --txt_lambda 0.2 --hidden_dim 128 --learning_rate 1e-4
python experiments.py --img_lambda 1 --txt_lambda 0 --hidden_dim 128 --learning_rate 1e-4