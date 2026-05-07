import os
import glob
import pickle
import pandas as pd
import json

# 解析 pkl 的根目录（即 sicnav 目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
sicnav_dir = os.path.dirname(current_dir)

def parse_pkl_files(results_dir=sicnav_dir):
    """遍历所有的 results_* 文件夹并解析 pkl 提取指标"""
    data_list = []
    
    # 找到所有的 pkl 文件 (假设路径是 results_<policy>/<env>/xxx.pkl)
    pkl_files = glob.glob(os.path.join(results_dir, "results_*", "**", "*.pkl"), recursive=True)
    
    for file_path in pkl_files:
        try:
            with open(file_path, 'rb') as f:
                summ_dict = pickle.load(f)
            
            # 使用os.sep将路径按系统分隔分开，匹配 policies 文件夹名
            path_parts = file_path.split(os.sep)
            policy_name = ""
            for part in path_parts:
                if part.startswith("results_"):
                    policy_name = part.replace("results_", "")
                    break
            
            record = {
                "Policy": policy_name,
                "Test_Case": summ_dict.get('test_case', -1),
                "Repeat_Idx": summ_dict.get('repeat_idx', -1),
                "Success": 1 if summ_dict.get('test_case_success') == 1 else 0,
                "Arrival_Time": summ_dict.get('nav_time', None),
                "Collision_Rate": summ_dict.get('coll_freq', 0.0),
                "Freezing_Rate": summ_dict.get('frozen_freq', 0.0),
                "Total_Steps": summ_dict.get('num_steps', 0)
            }
            data_list.append(record)
        except Exception as e:
            print(f"⚠️ 解析 {file_path} 失败: {e}")
            
    return pd.DataFrame(data_list)

def aggregate_and_save(df):
    """按策略计算平均值并保存"""
    summary = df.groupby('Policy').agg({
        'Success': 'mean',          # 平均成功率
        'Arrival_Time': 'mean',     # 平均到达时间
        'Collision_Rate': 'mean',   # 平均碰撞率
        'Freezing_Rate': 'mean'     # 平均冻结率
    }).reset_index()
    
    summary.rename(columns={
        'Success': 'Success Rate',
        'Arrival_Time': 'Avg Arrival Time',
        'Collision_Rate': 'Collision Rate',
        'Freezing_Rate': 'Freezing Rate'
    }, inplace=True)

    print("📊 实验数据汇总:")
    try:
        print(summary.to_markdown(index=False))
    except ImportError:
        print(summary.to_string(index=False))

    csv_path = os.path.join(current_dir, "summary_results.csv")
    json_path = os.path.join(current_dir, "detailed_results.json")

    # 保存为 CSV (最适合放进论文/Excel)
    summary.to_csv(csv_path, index=False)
    # 保存所有的详细记录为 JSON
    df.to_json(json_path, orient='records', indent=4)
    
    print(f"\n✅ 数据已经成功保存为 \n 1. {csv_path}\n 2. {json_path}")

if __name__ == "__main__":
    df_raw = parse_pkl_files()
    if df_raw.empty:
        print("未找到任何实验结果 .pkl 文件，请先运行实验！")
    else:
        aggregate_and_save(df_raw)
