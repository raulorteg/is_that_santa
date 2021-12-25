python create_db.py --path_imgs="datasets/train" --name_db="train_dataset.txt"
python create_db.py --path_imgs="datasets/test" --name_db="test_dataset.txt"
python main.py --n_epochs=100 --lr=0.0001 --batch_size=32 --latent_size=100 --resize=128 --val_freq=1
python create_plots.py