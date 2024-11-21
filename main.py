import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

colors = {
    "CC247C": {"hex": "#CC247C", "rgb": (204, 36, 124)},  # 紫红
    "E95351": {"hex": "#E95351", "rgb": (233, 83, 81)},   # 橙红
    "F7A24F": {"hex": "#F7A24F", "rgb": (247, 162, 79)},  # 橙黄
    "FBEB66": {"hex": "#FBEB66", "rgb": (251, 235, 102)}, # 黄
    "4EA660": {"hex": "#4EA660", "rgb": (78, 166, 96)},   # 绿
    "79CAFB": {"hex": "#79CAFB", "rgb": (121, 202, 251)}, # 浅蓝
    "5292F7": {"hex": "#5292F7", "rgb": (82, 146, 247)},  # 蓝
    "AA77E9": {"hex": "#AA77E9", "rgb": (170, 119, 233)}, # 紫
    "4484B1": {"hex": "#4484B1", "rgb": (68, 132, 177)},  # 深蓝
    "9BBAD6": {"hex": "#9BBAD6", "rgb": (155, 186, 214)}, # 灰蓝
    "F5A361": {"hex": "#F5A361", "rgb": (245, 163, 97)},  # 橙
    "F6BB8F": {"hex": "#F6BB8F", "rgb": (246, 184, 143)}, # 灰橙
}

# 读取CSV文件
file_path = r'F:\桥本研实验数据记录\training_log\Oct21_23-55-05_imitate\csv\train_mean_reward.csv'
df = pd.read_csv(file_path)

curves = {
    "Object speed: 5 m/s": {"x": df['Step'].tolist(), "y": df['Value'].tolist()},
}


def smooth_curve(data, window_size=50):
    smoothed_y = []
    for i in range(len(data)):
        if i == 0:
            # 第一个值为真实值
            smoothed_y.append(data[i])
        else:
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2)
            smoothed_y.append(sum(data[start:end]) / (end - start))
    return smoothed_y


for label, datas in curves.items():
    smoothed_values = smooth_curve(datas["y"], window_size=10)

    plt.plot(datas["x"], datas["y"], '-', color=colors["E95351"]["hex"], alpha=0.2)
    plt.plot(datas["x"], smoothed_values, '-', color=colors["E95351"]["hex"], label=label)


# 获取当前轴
ax = plt.gca()

# 隐藏顶部和右侧的边线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Training episodes')
plt.ylabel('Reward')
plt.legend(loc='lower right')
plt.show()

