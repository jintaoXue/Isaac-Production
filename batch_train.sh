#!/bin/bash

# 检查命令行参数
if [ $# -eq 0 ]; then
    echo "用法: $0 [A|B|1-10]"
    echo "  A: 运行A组训练 (1-5)"
    echo "  B: 运行B组训练 (6-10)"
    echo "  1-10: 运行单个训练序号"
    exit 1
fi

GROUP=$1

# 检查是否为数字（1-10）
if [[ "$GROUP" =~ ^[1-9]$|^10$ ]]; then
    echo "运行单个训练序号: $GROUP"
    SINGLE_TEST=true
else
    if [ "$GROUP" != "A" ] && [ "$GROUP" != "B" ]; then
        echo "错误: 参数必须是 A、B 或 1-10 中的数字"
        echo "用法: $0 [A|B|1-10]"
        exit 1
    fi
    SINGLE_TEST=false
fi

# 定义训练函数

run_test_2() {
    echo "运行训练 2: D3QN penalty"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate
}

run_test_1() {
    echo "运行训练 1: D3QN"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate
}

run_test_3() {
    echo "运行训练 3: PF-CD3Q"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --use_fatigue_mask
}

run_test_5() {
    echo "运行训练 5: DQN with penalty"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate 

}

run_test_6() {
    echo "运行训练 6: PF-DQN"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --use_fatigue_mask
}

run_test_7() {
    echo "运行训练 7: PPO-dis with penalty"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppo_dis --headless --wandb_activate
}

run_test_8() {
    echo "运行训练 8: PF-PPO-dis"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppo_dis --headless --wandb_activate --use_fatigue_mask
}

run_test_9() {
    echo "运行训练 9: PPO-lag"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppolag_filter_dis --headless --wandb_activate
}

run_test_10() {
    echo "运行训练 10: PF-PPO-lag"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppolag_filter_dis --headless --wandb_activate --use_fatigue_mask
}

# 单个训练
if [ "$SINGLE_TEST" = true ]; then
    case $GROUP in
        2) run_test_2 ;;
        3) run_test_3 ;;
        5) run_test_5 ;;
        6) run_test_6 ;;
        7) run_test_7 ;;
        8) run_test_8 ;;
        9) run_test_9 ;;
        10) run_test_10 ;;
        *) echo "错误: 无效的训练序号 $GROUP" ;;
    esac
    echo "训练 $GROUP 完成！"
    exit 0
fi

# A组训练 (1-5)
if [ "$GROUP" = "A" ]; then
    echo "=== 运行A组训练 (1-5) ==="
    run_test_2
    run_test_3
    run_test_5
    run_test_6  
    echo "A组训练完成！"
fi

# B组训练 (6-10)
if [ "$GROUP" = "B" ]; then
    echo "=== 运行B组训练 (6-10) ==="
    run_test_7
    run_test_8
    run_test_9
    run_test_10
    echo "B组训练完成！"
fi

echo "所有训练完成！"