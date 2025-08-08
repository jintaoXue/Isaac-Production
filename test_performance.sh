#!/bin/bash

# 检查命令行参数
if [ $# -eq 0 ]; then
    echo "用法: $0 [A|B|1-10]"
    echo "  A: 运行A组测试 (1-5)"
    echo "  B: 运行B组测试 (6-10)"
    echo "  1-10: 运行单个测试序号"
    exit 1
fi

GROUP=$1

# 检查是否为数字（1-10）
if [[ "$GROUP" =~ ^[1-9]$|^10$ ]]; then
    echo "运行单个测试序号: $GROUP"
    SINGLE_TEST=true
else
    if [ "$GROUP" != "A" ] && [ "$GROUP" != "B" ]; then
        echo "错误: 参数必须是 A、B 或 1-10 中的数字"
        echo "用法: $0 [A|B|1-10]"
        exit 1
    fi
    SINGLE_TEST=false
fi

# 定义测试函数
run_test_1() {
    echo "运行测试 1: D3QN nomask_4090_rl_filter_2025-07-25_15-02-16"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --test_all_settings --other_filters \
            --load_dir "/rl_filter_2025-07-25_15-02-16/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_2() {
    echo "运行测试 2: D3QN penalty_4070_rl_filter_2025-07-29_22-22-18"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --test_all_settings --other_filters \
            --load_dir "/rl_filter_2025-07-29_22-22-18/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_3() {
    echo "运行测试 3: PF-CD3Q 4070_rl_filter_2025-07-20_12-17-12"
    # list=(49600)
    list=(
        52800
        62400
        71600
        80400
    )
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --use_fatigue_mask --test_all_settings --other_filters \
            --load_dir "/rl_filter_2025-07-20_12-17-12/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_4() {
    echo "运行测试 4: mask_penalty_4090_rl_filter_2025-07-27_14-41-12"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --use_fatigue_mask --test_all_settings --other_filters \
            --load_dir "/rl_filter_2025-07-27_14-41-12/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_5() {
    echo "运行测试 5: DQN with penalty penalty_4070_dqn_2025-07-27_11-39-32"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --test --test_all_settings --other_filters \
            --load_dir "/dqn_2025-07-27_11-39-32/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_6() {
    echo "运行测试 6: PF-DQN 4090_dqn_2025-07-29_13-21-06"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --test --use_fatigue_mask --test_all_settings --other_filters \
            --load_dir "/dqn_2025-07-29_13-21-06/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_7() {
    echo "运行测试 7: PPO-dis with penalty 4070_penalty_ppo_dis_2025-07-31_13-37-58"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppo_dis --headless --wandb_activate --test --test_all_settings --other_filters \
            --load_dir "/ppo_dis_2025-07-31_13-37-58/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_8() {
    echo "运行测试 8: PF-PPO-dis 4090_ppo_dis_2025-07-30_13-18-07"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppo_dis --headless --wandb_activate --test --use_fatigue_mask --test_all_settings --other_filters \
            --load_dir "/ppo_dis_2025-07-30_13-18-07/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_9() {
    echo "运行测试 9: PPO-lag nomask_4070_ppolag_filter_dis_2025-07-23_22-24-04"
    list=(49600)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppolag_filter_dis --headless --wandb_activate --test --test_all_settings --other_filters \
            --load_dir "/ppolag_filter_dis_2025-07-23_22-24-04/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_10() {
    echo "运行测试 10: PF-PPO-lag 4070_ppolag_filter_dis_2025-07-21_23-34-32"
    list=(42800)
    for num in $list
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppolag_filter_dis --headless --wandb_activate --test --use_fatigue_mask --test_all_settings --other_filters \
            --load_dir "/ppolag_filter_dis_2025-07-21_23-34-32/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

# 单个测试
if [ "$SINGLE_TEST" = true ]; then
    case $GROUP in
        1) run_test_1 ;;
        2) run_test_2 ;;
        3) run_test_3 ;;
        4) run_test_4 ;;
        5) run_test_5 ;;
        6) run_test_6 ;;
        7) run_test_7 ;;
        8) run_test_8 ;;
        9) run_test_9 ;;
        10) run_test_10 ;;
        *) echo "错误: 无效的测试序号 $GROUP" ;;
    esac
    echo "测试 $GROUP 完成！"
    exit 0
fi

# A组测试 (1-5)
if [ "$GROUP" = "A" ]; then
    echo "=== 运行A组测试 (1-5) ==="
    run_test_1
    run_test_2
    run_test_3
    run_test_4
    run_test_5
    echo "A组测试完成！"
fi

# B组测试 (6-10)
if [ "$GROUP" = "B" ]; then
    echo "=== 运行B组测试 (6-10) ==="
    run_test_6
    run_test_7
    run_test_8
    run_test_9
    run_test_10
    echo "B组测试完成！"
fi

echo "所有测试完成！"