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
file_path = r'F:\桥本研实验数据记录\training_log\mid_term_data\ad.csv'
ad_df = pd.read_csv(file_path)

file_path = r'F:\桥本研实验数据记录\training_log\mid_term_data\re.csv'
re_df = pd.read_csv(file_path)

# curves = {
#     "Ad-1, Ball speed: 5 m/s": {"x": ad_1_df['Step'].tolist(), "y": ad_1_df['Value'].tolist(), "file_start": 0, "start": 0, "end": 10000},
#     "Ad-2, Ball speed: 10 m/s": {"x": ad_2_df['Step'].tolist(), "y": ad_2_df['Value'].tolist(), "file_start": 6000, "start": 10000, "end": 13000},
#     "Re-1, Ball speed: 5 m/s": {"x": re_1_df['Step'].tolist(), "y": re_1_df['Value'].tolist(), "start": 0, "end": 10000},
#     "Re-2, Ball speed: 10 m/s": {"x": re_2_df['Step'].tolist(), "y": re_2_df['Value'].tolist(), "start": 10000, "end": 13000},
# }


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


datas = ad_df['Value'].tolist()
x = np.array(ad_df['Step'].tolist())
smoothed_values = smooth_curve(datas, window_size=10)
devide = 1000
plt.plot(x[:devide], datas[:devide], '-', color=colors["E95351"]["hex"], alpha=0.2)
plt.plot(x[:devide], smoothed_values[:devide], '-', color=colors["E95351"]["hex"], label="Ad-1, Ball speed: 5 m/s")

plt.plot(x[devide:], datas[devide:], '-', color=colors["F7A24F"]["hex"], alpha=0.2)
plt.plot(x[devide:], smoothed_values[devide:], '-', color=colors["F7A24F"]["hex"], label="Ad-2, Ball speed: 10 m/s")

# datas = re_df['Value'].tolist()
# x = np.array(re_df['Step'].tolist())
# smoothed_values = smooth_curve(datas, window_size=10)
# devide = 1000
# plt.plot(x[:devide], datas[:devide], '-', color=colors["5292F7"]["hex"], alpha=0.2)
# plt.plot(x[:devide], smoothed_values[:devide], '-', color=colors["5292F7"]["hex"], label="Re-1, Ball speed: 5 m/s")
#
# smoothed_values = smooth_curve(datas, window_size=10)
# plt.plot(x[devide:], datas[devide:], '-', color=colors["AA77E9"]["hex"], alpha=0.2)
# plt.plot(x[devide:], smoothed_values[devide:], '-', color=colors["AA77E9"]["hex"], label="Re-2, Ball speed: 10 m/s")


# 获取当前轴
ax = plt.gca()

# 隐藏顶部和右侧的边线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Training episodes', fontsize=18)
plt.ylabel('Reward', fontsize=18)
plt.ylim(-15, 3)
plt.legend(loc='lower right', fontsize=18)
plt.show()

