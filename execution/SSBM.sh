cd ../src

# ../../parallel -j4 --resume-failed --results ../Output/SSBM1000_1 --joblog ../joblog/SSBM1000_joblog_1 CUDA_VISIBLE_DEVICES=0 python ./train.py --dataset SSBM --N 1000 --size_ratio 1.5 --p 0.01 --K 20  --eta {1}  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3

../../parallel -j3 --resume-failed --results ../Output/SSBM1000_2 --joblog ../joblog/SSBM1000_joblog_2 CUDA_VISIBLE_DEVICES=0 python ./train.py --dataset SSBM --N 1000 --size_ratio 1.5 --p 0.01 --K 5  --eta {1}  ::: 0 0.05 0.1 0.15 0.2 0.25

# ../../parallel -j5 --resume-failed --results ../Output/0512SSBM1000_3 --joblog ../joblog/0512SSBM1000_3_joblog CUDA_VISIBLE_DEVICES=0 python ./train.py --dataset SSBM --N 1000 --size_ratio 2 --p 0.1 --K 2  --eta {1}  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35

../../parallel -j1 --resume-failed --results ../Output/SSBM5000_1 --joblog ../joblog/SSBM5000_joblog_1 CUDA_VISIBLE_DEVICES=0 python ./train.py --dataset SSBM --N 5000 --size_ratio 1.5 --p 0.01 --K 5 --eta {1}  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4

../../parallel -j1 --resume-failed --results ../Output/SSBM10000 --joblog ../joblog/SSBM10000_joblog CUDA_VISIBLE_DEVICES=0 python ./train.py --dataset SSBM --N 10000 --size_ratio 1.5 --p 0.01 --K 5  --eta {1} ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4

../../parallel -j1 --resume-failed --results ../Output/SSBM30000 --joblog ../joblog/SSBM30000_joblog CUDA_VISIBLE_DEVICES=0 python ./train.py --dataset SSBM --N 30000 --size_ratio 1.5 --p 0.001 --K 5  --eta {1}  ::: 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4