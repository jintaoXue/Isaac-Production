#!/usr/bin/env python3
"""
测试误差放大功能的脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from pf_filter import ParticleFilter

def test_error_amplification():
    """测试误差放大功能"""
    
    # 创建测试误差数据
    errors = np.array([0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    
    # 创建粒子滤波器实例
    pf = ParticleFilter(dt=0.1, num_steps=10, true_lambda=0.5, F0=0.0, 
                       num_particles=8, sigma_w=0.001, sigma_v=0.001,
                       lamda_init=0.5, upper_bound=2.0, lower_bound=0.0,
                       error_amplification_alpha=3.0)
    
    print("原始误差:", errors)
    print("误差范围: min={:.3f}, max={:.3f}".format(np.min(errors), np.max(errors)))
    
    # 测试不同的放大方法
    methods = ['exponential', 'power', 'sigmoid', 'linear']
    
    plt.figure(figsize=(15, 10))
    
    for i, method in enumerate(methods):
        amplified = pf.amplify_errors(errors, method=method)
        
        plt.subplot(2, 2, i+1)
        plt.plot(errors, amplified, 'o-', linewidth=2, markersize=8)
        plt.xlabel('误差', fontsize=12)
        plt.ylabel('放大后的权重', fontsize=12)
        plt.title(f'{method} 方法 (α={pf.error_amplification_alpha})', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标注
        for j, (err, amp) in enumerate(zip(errors, amplified)):
            plt.annotate(f'{amp:.3f}', (err, amp), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('error_amplification_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细结果
    print("\n误差放大效果对比:")
    print("-" * 60)
    for method in methods:
        amplified = pf.amplify_errors(errors, method=method)
        print(f"{method:12}: {amplified}")
        print(f"{'':12}  权重和: {np.sum(amplified):.6f}")

def test_parameter_sensitivity():
    """测试参数敏感性"""
    
    errors = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    
    # 测试不同的alpha值
    alphas = [1.0, 2.0, 3.0, 5.0, 10.0]
    
    plt.figure(figsize=(15, 10))
    
    for i, alpha in enumerate(alphas):
        pf = ParticleFilter(dt=0.1, num_steps=10, true_lambda=0.5, F0=0.0, 
                           num_particles=8, sigma_w=0.001, sigma_v=0.001,
                           lamda_init=0.5, upper_bound=2.0, lower_bound=0.0,
                           error_amplification_alpha=alpha)
        
        amplified = pf.amplify_errors(errors, method='exponential')
        
        plt.subplot(2, 3, i+1)
        plt.plot(errors, amplified, 'o-', linewidth=2, markersize=8)
        plt.xlabel('误差', fontsize=10)
        plt.ylabel('权重', fontsize=10)
        plt.title(f'α = {alpha}', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n参数敏感性测试:")
    print("-" * 40)
    for alpha in alphas:
        pf = ParticleFilter(dt=0.1, num_steps=10, true_lambda=0.5, F0=0.0, 
                           num_particles=8, sigma_w=0.001, sigma_v=0.001,
                           lamda_init=0.5, upper_bound=2.0, lower_bound=0.0,
                           error_amplification_alpha=alpha)
        amplified = pf.amplify_errors(errors, method='exponential')
        print(f"α = {alpha:2.1f}: 最小权重 = {np.min(amplified):.6f}, 最大权重 = {np.max(amplified):.6f}")

if __name__ == "__main__":
    print("测试误差放大功能...")
    test_error_amplification()
    
    print("\n" + "="*60)
    print("测试参数敏感性...")
    test_parameter_sensitivity()
    
    print("\n测试完成！") 