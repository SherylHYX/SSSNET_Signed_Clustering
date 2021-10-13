cd ../src

../../parallel -j3 --resume-failed --results ../Output/polarized1050 --joblog ../joblog/polarized1050_joblog CUDA_VISIBLE_DEVICES=0 python ./train.py --dataset polarized --N 200 --total_n 1050 --size_ratio 1 --p 0.1 --K 2 --num_com 2  --eta {1}  ::: 0 0.05 0.1 0.15 0.2 0.25

../../parallel -j2 --resume-failed --results ../Output/polarized5000_1 --joblog ../joblog/polarized5000_joblog_1 CUDA_VISIBLE_DEVICES=0 python ./train.py --dataset polarized --N 500 --total_n 5000 --size_ratio 1.5 --p 0.1 --K 2 --num_com 3  --eta {1}  ::: 0 0.05 0.1 0.15 0.2

../../parallel -j2 --resume-failed --results ../Output/polarized5000_2 --joblog ../joblog/polarized5000_joblog_2 CUDA_VISIBLE_DEVICES=0 python ./train.py --dataset polarized --N 500 --total_n 5000 --size_ratio 1.5 --p 0.1 --K 2 --num_com 5  --eta {1}  ::: 0 0.05 0.1 0.15 0.2 0.25

../../parallel -j1 --resume-failed --results ../Output/polarized10000_1 --joblog ../joblog/polarized10000_joblog_1 CUDA_VISIBLE_DEVICES=0 python ./train.py --dataset polarized --N 2000 --total_n 10000 --size_ratio 1.5 --p 0.01 --K 2 --num_com 2   --eta {1}  ::: 0 0.05 0.1 0.15 0.2 0.25