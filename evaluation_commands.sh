# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate rl

# cd fastar

# German Credit
python -W ignore main.py --algo ppo --use-gae --lr 0.0001 --clip-param 0.2 --value-loss-coef 0.5 --num-processes 1 --num-steps 256 --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir "output/trained_models/german5_sampletrain_search_01_0.99_256_0.001_0.2_nr" --env-name "gym_midline:german-nr-v01" --num-env-steps 5000000 --save-interval 1000 --gamma 0.99 --log-dir "output/log/german_nr" --no-cuda --eval

# Adult Income
python -W ignore main.py --algo ppo --use-gae --lr 0.0001 --clip-param 0.2 --value-loss-coef 0.5 --num-processes 1 --num-steps 256 --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir "output/trained_models/adult_sampletrain_search_01_0.99_256_0.0001_0.2_nr" --env-name "gym_midline:adult-nr-v01" --num-env-steps 10000000 --save-interval 5000 --gamma 0.99 --log-dir "output/log/adult_nr" --eval

# COMPAS
python -W ignore main.py --algo ppo --use-gae --lr 0.0001 --clip-param 0.2 --value-loss-coef 0.5 --num-processes 1 --num-steps 256 --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir "output/trained_models/compas_sampletrain_search_01_0.99_256_0.0001_0.2_nr" --env-name "gym_midline:compas-v01" --num-env-steps 5000000 --save-interval 4000 --gamma 0.99 --log-dir "output/log/compas_nr" --eval