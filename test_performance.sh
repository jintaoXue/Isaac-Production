# #！/bin/bash
# load_dir="/rl_filter_2025-07-20_12-17-12/nn"
# relative_pth="/logs/rl_games/HRTA_direct"
# str="/"
# work_space_path=$(pwd)
# dir_path=$work_space_path$relative_pth$load_dir
# # path=$1
# files=$(ls $dir_path)



############### Part 1 in workstation 4090 ###############
list=(
 23700
 22900
 24800
 18100
)

for num in $list
do
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --test \
        --load_dir "/dqn_2025-07-29_13-21-06/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
done

# 1. D3QN nomask_4090_rl_filter_2025-07-25_15-02-16
list=(
 23700
 22900
 24800
 18100
)

for num in $list
do
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --test \
        --load_dir "/dqn_2025-07-29_13-21-06/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
done




############### Part 2 in workstation 4070 ###############
list=(
 23700
 22900
 24800
 18100
)

for num in $list
do
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --test \
        --load_dir "/dqn_2025-07-29_13-21-06/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
done