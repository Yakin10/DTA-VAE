# DTA-VAE
DTA+VAE:Drug Target Affinity prediction with SELFIES String via variational autoencoder and Transformer6 protein model

# Usage
```
pip install git+https://github.com/facebookresearch/esm.git

python run_experiments.py --num_windows 32 \
                          --seq_window_lengths 8 12 \
                          --smi_window_lengths 4 8 \
                          --batch_size 256 \
                          --num_epoch 100 \
                          --max_seq_len 1000 \
                          --max_smi_len 100 \
                          --dataset_path 'data/kiba/' \
                          --problem_type 1 \
                          --log_dir 'logs/'


```
