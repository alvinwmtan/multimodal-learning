#!usr/bin/bash

python experiments.py --img_lambda 0.4 --txt_lambda 0.4 --hidden_dim 128 --learning_rate 1e-4 --num_classes 6 --data_path frame-extraction/all_namings_cleaned_multi.csv --multi_mode
python experiments.py --img_lambda 0.4 --txt_lambda 0.6 --hidden_dim 128 --learning_rate 1e-4 --num_classes 6 --data_path frame-extraction/all_namings_cleaned_multi.csv --multi_mode
python experiments.py --img_lambda 0.6 --txt_lambda 0.4 --hidden_dim 128 --learning_rate 1e-4 --num_classes 6 --data_path frame-extraction/all_namings_cleaned_multi.csv --multi_mode
python experiments.py --img_lambda 0.2 --txt_lambda 0.2 --hidden_dim 128 --learning_rate 1e-4 --num_classes 6 --data_path frame-extraction/all_namings_cleaned_multi.csv --multi_mode
python experiments.py --img_lambda 0.2 --txt_lambda 0.4 --hidden_dim 128 --learning_rate 1e-4 --num_classes 6 --data_path frame-extraction/all_namings_cleaned_multi.csv --multi_mode
python experiments.py --img_lambda 0.2 --txt_lambda 0.6 --hidden_dim 128 --learning_rate 1e-4 --num_classes 6 --data_path frame-extraction/all_namings_cleaned_multi.csv --multi_mode
python experiments.py --img_lambda 0.2 --txt_lambda 0.8 --hidden_dim 128 --learning_rate 1e-4 --num_classes 6 --data_path frame-extraction/all_namings_cleaned_multi.csv --multi_mode
python experiments.py --img_lambda 0.4 --txt_lambda 0.2 --hidden_dim 128 --learning_rate 1e-4 --num_classes 6 --data_path frame-extraction/all_namings_cleaned_multi.csv --multi_mode
python experiments.py --img_lambda 0.6 --txt_lambda 0.2 --hidden_dim 128 --learning_rate 1e-4 --num_classes 6 --data_path frame-extraction/all_namings_cleaned_multi.csv --multi_mode
python experiments.py --img_lambda 0.8 --txt_lambda 0.2 --hidden_dim 128 --learning_rate 1e-4 --num_classes 6 --data_path frame-extraction/all_namings_cleaned_multi.csv --multi_mode