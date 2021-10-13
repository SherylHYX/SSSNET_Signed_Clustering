cd ../src

python train.py --dataset sampson -SP --feature_options given None --directed --seed_ratio 0.5 --test_ratio 0.5 --no_validation --samples 200 --num_trials 10 --seeds 31 --epochs 80
python train.py --dataset rainfall -SP --seed_ratio 0.1 --dense --feature_options L --no_validation --num_trials 10 --seeds 31
python train.py --dataset SP1500 -SP --seed_ratio 0.1 --dense --feature_options L --no_validation --num_trials 10 --seeds 31

python train.py -All --dataset PPI -SP --seed_ratio 0 --feature_options L --num_trials 10 --seeds 31
python train.py -All --dataset wikirfa -SP --seed_ratio 0 --directed --feature_options L --num_trials 10 --seeds 31
../../parallel -j3 --resume-failed --results ../Output/yearly --joblog ../joblog/yearly_joblog CUDA_VISIBLE_DEVICES=0 python train.py --year_index {1} --dataset MR_yearly -SP --seed_ratio 0.1 --dense --feature_options given L --no_validation --num_trials 10 --seeds 31 ::: {0..20}

