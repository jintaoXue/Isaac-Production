#!/bin/bash

set -euo pipefail

FTG_VALUES=($(python - <<'PY'
vals = [f"{i/10:.1f}" for i in range(11)]
print(" ".join(vals))
PY
))

run_test_3() {
    local ftg_value=$1
    local checkpoints=(49600)
    echo "运行测试 3: PF-CD3Q | ftg_thresh_phy=${ftg_value}"
    for num in "${checkpoints[@]}"; do
        python train.py \
            --task Isaac-TaskAllocation-Direct-v1 \
            --algo rl_filter \
            --headless \
            --wandb_activate \
            --test \
            --use_fatigue_mask \
            --test_all_settings \
            --other_filters \
            --load_dir "/rl_filter_2025-07-20_12-17-12/nn" \
            --load_name "/HRTA_direct_ep_${num}.pth" \
            --wandb_project test_ftg_sensitivity \
            --test_times 1 \
            --ftg_thresh_phy "${ftg_value}"
    done
}

run_test_9() {
    local ftg_value=$1
    local checkpoints=(49600)
    echo "运行测试 9: PPO-lag | ftg_thresh_phy=${ftg_value}"
    for num in "${checkpoints[@]}"; do
        python train.py \
            --task Isaac-TaskAllocation-Direct-v1 \
            --algo ppolag_filter_dis \
            --headless \
            --wandb_activate \
            --test \
            --test_all_settings \
            --other_filters \
            --load_dir "/ppolag_filter_dis_2025-08-08_13-49-16/nn" \
            --load_name "/HRTA_direct_ep_${num}.pth" \
            --wandb_project test_ftg_sensitivity \
            --test_times 1 \
            --ftg_thresh_phy "${ftg_value}"
    done
}

for ftg in "${FTG_VALUES[@]}"; do
    echo "===== 开始疲劳阈值 ${ftg} ====="
    run_test_3 "${ftg}"
    run_test_9 "${ftg}"
    echo "===== 完成疲劳阈值 ${ftg} ====="
done

echo "所有疲劳敏感性测试完成！"