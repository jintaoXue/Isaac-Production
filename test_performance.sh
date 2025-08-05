#!/bin/bash

# 检查命令行参数
if [ $# -eq 0 ]; then
    echo "用法: $0 [A|B]"
    echo "  A: 运行A组测试 (1-5)"
    echo "  B: 运行B组测试 (6-10)"
    exit 1
fi

GROUP=$1

if [ "$GROUP" != "A" ] && [ "$GROUP" != "B" ]; then
    echo "错误: 参数必须是 A 或 B"
    echo "用法: $0 [A|B]"
    exit 1
fi

echo "运行 $GROUP 组测试..."

# A组测试 (1-5)
if [ "$GROUP" = "A" ]; then
    echo "=== 运行A组测试 (1-5) ==="
    
    ###1. D3QN nomask_4090_rl_filter_2025-07-25_15-02-16
    echo "运行测试 1: D3QN nomask_4090_rl_filter_2025-07-25_15-02-16"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --test_all_settings \
            --load_dir "/rl_filter_2025-07-25_15-02-16/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
    done

    ###2. D3QN penalty_4070_rl_filter_2025-07-29_22-22-18
    echo "运行测试 2: D3QN penalty_4070_rl_filter_2025-07-29_22-22-18"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --test_all_settings \
            --load_dir "/rl_filter_2025-07-29_22-22-18/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
    done

    ###3. PF-CD3Q 4070_rl_filter_2025-07-20_12-17-12
    echo "运行测试 3: PF-CD3Q 4070_rl_filter_2025-07-20_12-17-12"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --use_fatigue_mask --test_all_settings \
            --load_dir "/rl_filter_2025-07-20_12-17-12/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
    done

    ###4. mask_penalty_4090_rl_filter_2025-07-27_14-41-12
    echo "运行测试 4: mask_penalty_4090_rl_filter_2025-07-27_14-41-12"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --use_fatigue_mask --test_all_settings \
            --load_dir "/rl_filter_2025-07-27_14-41-12/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
    done

    ###5. DQN with penalty penalty_4070_dqn_2025-07-27_11-39-32
    echo "运行测试 5: DQN with penalty penalty_4070_dqn_2025-07-27_11-39-32"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --test --test_all_settings \
            --load_dir "/dqn_2025-07-27_11-39-32/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
    done

    echo "A组测试完成！"
fi

# B组测试 (6-10)
if [ "$GROUP" = "B" ]; then
    echo "=== 运行B组测试 (6-10) ==="
    
    ###6. PF-DQN 4090_dqn_2025-07-29_13-21-06
    echo "运行测试 6: PF-DQN 4090_dqn_2025-07-29_13-21-06"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --test --use_fatigue_mask --test_all_settings \
            --load_dir "/dqn_2025-07-29_13-21-06/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
    done

    ###7. PPO-dis with penalty 4070_penalty_ppo_dis_2025-07-31_13-37-58
    echo "运行测试 7: PPO-dis with penalty 4070_penalty_ppo_dis_2025-07-31_13-37-58"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppo_dis --headless --wandb_activate --test --test_all_settings \
            --load_dir "/ppo_dis_2025-07-31_13-37-58/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
    done

    ###8. PF-PPO-dis 4090_ppo_dis_2025-07-30_13-18-07
    echo "运行测试 8: PF-PPO-dis 4090_ppo_dis_2025-07-30_13-18-07"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppo_dis --headless --wandb_activate --test --use_fatigue_mask --test_all_settings \
            --load_dir "/ppo_dis_2025-07-30_13-18-07/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
    done

    ###9. PPO-lag nomask_4070_ppolag_filter_dis_2025-07-23_22-24-04
    echo "运行测试 9: PPO-lag nomask_4070_ppolag_filter_dis_2025-07-23_22-24-04"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppolag_filter_dis --headless --wandb_activate --test --test_all_settings \
            --load_dir "/ppolag_filter_dis_2025-07-23_22-24-04/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
    done

    ###10. PF-PPO-lag 4070_ppolag_filter_dis_2025-07-21_23-34-32
    echo "运行测试 10: PF-PPO-lag 4070_ppolag_filter_dis_2025-07-21_23-34-32"
    list=(42800)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppolag_filter_dis --headless --wandb_activate --test --use_fatigue_mask --test_all_settings \
            --load_dir "/ppolag_filter_dis_2025-07-21_23-34-32/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 1
    done

    echo "B组测试完成！"
fi

echo "所有测试完成！"