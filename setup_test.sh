cd ./src/
python train.py -D --regenerate_data --dataset polarized --N 200 --all_methods SSSNET
python train.py -D -SP --no-cuda
python train.py -D --feature_options A_reg None
python train.py -D -All --dataset SP1500 -SP --dense --feature_options L None
python train.py -D --seed_ratio 0 --dataset sampson -SP --directed --feature_options given None
python train.py -D --seed_ratio 0.5 --load_only