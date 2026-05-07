import subprocess
import os

# 获取此脚本所在的目录，并计算 sicnav 的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sicnav_dir = os.path.dirname(current_dir)

# 配置实验参数
policies = ['ponav', 'dwa', 'sfm']  # 你需要对比的策略名称
test_cases = [0, 1, 2]              # 要测试的场景编号
repeats = 50                        # 每个场景测试的回合数

def run_experiments():
    print("🚀 开始自动批量运行实验...")
    # 切换到 sicnav 目录以确保 aggregate_results.py 能正常运行
    os.chdir(sicnav_dir)
    for policy in policies:
        for tc in test_cases:
            print(f"[{policy}] 正在运行 Test Case {tc} ({repeats} repeats)...")
            cmd = [
                "python", "aggregate_results.py",
                "--policy", policy,
                "--test_case", str(tc),
                "--repeat", str(repeats),
                # 若带视觉界面极慢，确保去掉 --visualize
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ 运行 {policy} 的 Test Case {tc} 时出错: {e}")

if __name__ == "__main__":
    run_experiments()
    print("✅ 所有实验运行完毕！")
