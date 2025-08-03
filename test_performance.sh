#！/bin/bash
load_dir="/rl_filter_2025-07-20_12-17-12/nn"
relative_pth="/logs/rl_games/HRTA_direct"
str="/"
work_space_path=$(pwd)
dir_path=$work_space_path$relative_pth$load_dir
# path=$1
files=$(ls $dir_path)

list=(
    1
    # 2
    # 3
    # 4
    # 5
)


python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --test --use_fatigue_mask \
    --load_dir "/dqn_2025-07-29_13-21-06/nn" --load_name "/HRTA_direct_ep_84400.pth" --wandb_project test_HRTA_fatigue --test_times 1




# for num in $list
# do
#     python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test \
#         --load_dir "$load_dir" --load_name "/HRTA_direct_ep_82400.pth" --wandb_project test_HRTA_fatigue --test_times 10
# #    echo $filename >> filename.txt
# #    echo -e >> filename.txt
# done

