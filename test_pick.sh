#!/usr/bin/env bash
set -euo pipefail

# 用法: bash test_pick.sh [模型目录]
# 功能: 提取目录下所有 HRTA_direct_ep_*.pth 的编号, 过滤 <15000 以及已完成编号, 并逐个运行测试

# 默认目录(可由命令行参数覆盖)
DEFAULT_DIR="/home/xue/work/Isaac-Production/logs/rl_games/HRTA_direct/rl_filter_2025-07-20_12-17-12/nn"
DIR=${1:-$DEFAULT_DIR}

# 已测试过的模型编号(填入数字, 用空格分隔)
done_list=(
    49600
    44800
    46800
    52400
    54400
    54000
    53600
    53200
    52800
    52400
    52000
    51600
    51200
    50800
    50400
)

if [ ! -d "$DIR" ]; then
    echo "错误: 目录不存在: $DIR"
    exit 1
fi

echo "扫描目录: $DIR"

# 收集所有符合命名的模型编号
mapfile -t all_nums < <(\
    find "$DIR" -maxdepth 1 -type f -name 'HRTA_direct_ep_*.pth' \
    | sed -E 's/.*HRTA_direct_ep_([0-9]+)\.pth/\1/' \
    | sort -n | uniq
)

if [ ${#all_nums[@]} -eq 0 ]; then
    echo "未找到任何模型文件: HRTA_direct_ep_*.pth"
    exit 0
fi

# 过滤 <15000
filtered_nums=()
for n in "${all_nums[@]}"; do
    if [ "$n" -ge 15000 ]; then
        filtered_nums+=("$n")
    fi
done

# 过滤 done_list 中的编号
is_done() {
    local val=$1
    for d in "${done_list[@]:-}"; do
        if [ "$d" = "$val" ]; then
            return 0
        fi
    done
    return 1
}

to_run=()
for n in "${filtered_nums[@]}"; do
    if ! is_done "$n"; then
        to_run+=("$n")
    fi
done

if [ ${#to_run[@]} -eq 0 ]; then
    echo "没有需要运行的模型编号 (均已过滤或已完成)"
    exit 0
fi

echo "将要运行的模型编号: ${to_run[*]}"

# 逐个运行
for num in "${to_run[@]}"; do
    echo "运行模型: HRTA_direct_ep_${num}.pth"
    python train.py \
        --task Isaac-TaskAllocation-Direct-v1 \
        --algo rl_filter \
        --headless \
        --wandb_activate \
        --test \
        --test_all_settings \
        --other_filters \
        --load_dir "/rl_filter_2025-07-20_12-17-12/nn" \
        --load_name "/HRTA_direct_ep_${num}.pth" \
        --wandb_project test_HRTA_fatigue \
        --test_times 50
done

echo "全部运行完成。"
