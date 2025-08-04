# #！/bin/bash
# load_dir="/rl_filter_2025-07-20_12-17-12/nn"
# relative_pth="/logs/rl_games/HRTA_direct"
# str="/"
# work_space_path=$(pwd)
# dir_path=$work_space_path$relative_pth$load_dir
# # path=$1
# files=$(ls $dir_path)


###1. D3QN nomask_4090_rl_filter_2025-07-25_15-02-16
list=(
49600
)

for num in $list
do
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test \
        --load_dir "/rl_filter_2025-07-25_15-02-16/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
done


############### 2. D3QN penalty_4070_rl_filter_2025-07-29_22-22-18 ###############
list=(
49600
)

for num in $list
do
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test \
        --load_dir "/rl_filter_2025-07-29_22-22-18/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
done

############### 3. PF-CD3Q 4070_rl_filter_2025-07-20_12-17-12 ###############
list=(
49600
)

for num in $list
do
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --use_fatigue_mask \
        --load_dir "/rl_filter_2025-07-20_12-17-12/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
done


############### 4. mask_penalty_4090_rl_filter_2025-07-27_14-41-12 ###############
list=(
49600
)

for num in $list
do
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --use_fatigue_mask \
        --load_dir "/rl_filter_2025-07-27_14-41-12/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
done


############### 5. DQN with penalty penalty_4070_dqn_2025-07-27_11-39-32 ###############
list=(
49600
)

for num in $list
do
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --test \
        --load_dir "/dqn_2025-07-27_11-39-32/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
done

############### 6. PF-DQN 4090_dqn_2025-07-29_13-21-06 ###############
list=(
49600
)

for num in $list
do
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --test --use_fatigue_mask\
        --load_dir "/dqn_2025-07-29_13-21-06/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
done    

############### 7. PPO-dis with penalty 4070_penalty_ppo_dis_2025-07-31_13-37-58 ###############
list=(
49600
)

for num in $list
do
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppo_dis --headless --wandb_activate --test \
        --load_dir "/ppo_dis_2025-07-31_13-37-58/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
done

############### 8. PF-PPO-dis 4090_ppo_dis_2025-07-30_13-18-07 ###############
list=(
49600
)

for num in $list
do
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppo_dis --headless --wandb_activate --test --use_fatigue_mask\
        --load_dir "/ppo_dis_2025-07-30_13-18-07/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
done

############### 9. PPO-lag nomask_4070_ppolag_filter_dis_2025-07-23_22-24-04 ###############
list=(
49600
)

for num in $list
do
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppolag_filter_dis --headless --wandb_activate --test \
        --load_dir "/ppolag_filter_dis_2025-07-23_22-24-04/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
done

############### 10. PF-PPO-lag 4070_ppolag_filter_dis_2025-07-21_23-34-32 ###############
list=(
42800
)

for num in $list
do
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppolag_filter_dis --headless --wandb_activate --test --use_fatigue_mask\
        --load_dir "/ppolag_filter_dis_2025-07-21_23-34-32/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
done