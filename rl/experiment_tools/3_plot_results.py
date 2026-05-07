import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# 论文级别的图表画风配置
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

def plot_metrics():
    csv_file = os.path.join(current_dir, "summary_results.csv")
    if not os.path.exists(csv_file):
        print(f"❌ 找不到 {csv_file}，请运行 '2_parse_results.py'！")
        return
        
    df = pd.read_csv(csv_file)
    
    metrics = ['Success Rate', 'Collision Rate', 'Freezing Rate', 'Avg Arrival Time']
    titles = ['Navigation Success Rate', 'Collision Rate', 'Freezing Rate', 'Average Arrival Time (s)']
    y_labels = ['Rate', 'Rate', 'Rate', 'Time (s)']
    colors = sns.color_palette("muted")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, y_labels)):
        ax = axes[idx]
        
        # 绘制柱状图
        sns.barplot(x='Policy', y=metric, data=df, ax=ax, palette=colors)
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Policy')
        
        # 给成功率固定Y轴为 0-1
        if 'Rate' in metric:
            ax.set_ylim([0, 1.05])
            
    plt.tight_layout()
    
    pdf_path = os.path.join(current_dir, 'experiment_metrics_plot.pdf')
    png_path = os.path.join(current_dir, 'experiment_metrics_plot.png')

    # 导出为 PDF (推荐，放大不失真) 和 PNG
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    print(f"📈 画图完毕！已保存至：\n- {pdf_path}\n- {png_path}")
    # plt.show() # 在无GUI环境下可能无法显示，可只写入文件

if __name__ == "__main__":
    plot_metrics()