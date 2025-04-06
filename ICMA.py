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


file_path = r'F:\桥本研实验数据记录\training_log\icma\mean_reward1.csv'
ad_df = pd.read_csv(file_path)
file_path = r'F:\桥本研实验数据记录\training_log\icma\mean_reward2.csv'
re_df = pd.read_csv(file_path)
datas = ad_df['Value'].tolist()
x = np.array(ad_df['Step'].tolist())
smoothed_values = smooth_curve(datas, window_size=10)
plt.plot(x, datas, '-', color=colors["E95351"]["hex"], alpha=0.2)
plt.plot(x, smoothed_values, '-', color=colors["E95351"]["hex"], label="Model-1, Ball speed: 5 m/s")

datas = re_df['Value'].tolist()
x = np.array(re_df['Step'].tolist())
smoothed_values = smooth_curve(datas, window_size=10)
plt.plot(x, datas, '-', color=colors["F7A24F"]["hex"], alpha=0.2)
plt.plot(x, smoothed_values, '-', color=colors["F7A24F"]["hex"], label="Model-2, Ball speed: 10 m/s")

ax = plt.gca()
plt.xlim(0, 13000)
plt.ylim(-5, 6)
plt.legend(loc='lower right', fontsize=12)
plt.show()

# file_path = r'F:\桥本研实验数据记录\training_log\icma\mean_event_recall_rate1.csv'
# mean_event_recall_rate1 = pd.read_csv(file_path)
#
# file_path = r'F:\桥本研实验数据记录\training_log\icma\mean_event_recall_rate2.csv'
# mean_event_recall_rate2 = pd.read_csv(file_path)
#
# file_path = r'F:\桥本研实验数据记录\training_log\icma\mean_event_success_rate1.csv'
# mean_event_success_rate1 = pd.read_csv(file_path)
#
# file_path = r'F:\桥本研实验数据记录\training_log\icma\mean_event_success_rate2.csv'
# mean_event_success_rate2 = pd.read_csv(file_path)
#
# datas = mean_event_success_rate1['Value'].tolist()
# x = np.array(mean_event_success_rate1['Step'].tolist())
# smoothed_values = smooth_curve(datas, window_size=10)
# plt.plot(x, datas, '-', color=colors["5292F7"]["hex"], alpha=0.2)
# plt.plot(x, smoothed_values, '-', color=colors["5292F7"]["hex"], label="Model-1, Rate1")
#
# datas = mean_event_recall_rate1['Value'].tolist()
# x = np.array(mean_event_recall_rate1['Step'].tolist())
# smoothed_values = smooth_curve(datas, window_size=10)
# plt.plot(x, datas, '-', color=colors["E95351"]["hex"], alpha=0.2)
# plt.plot(x, smoothed_values, '-', color=colors["E95351"]["hex"], label="Model-1, Rate2")
#
# datas = mean_event_success_rate2['Value'].tolist()
# x = np.array(mean_event_success_rate2['Step'].tolist())
# smoothed_values = smooth_curve(datas, window_size=10)
# plt.plot(x, datas, '-', color=colors["AA77E9"]["hex"], alpha=0.2)
# plt.plot(x, smoothed_values, '-', color=colors["AA77E9"]["hex"], label="Model-2, Rate1")
#
# datas = mean_event_recall_rate2['Value'].tolist()
# x = np.array(mean_event_recall_rate2['Step'].tolist())
# smoothed_values = smooth_curve(datas, window_size=10)
# plt.plot(x, datas, '-', color=colors["F7A24F"]["hex"], alpha=0.2)
# plt.plot(x, smoothed_values, '-', color=colors["F7A24F"]["hex"], label="Model-2, Rate2")


#ax = plt.gca()

# 隐藏顶部和右侧的边线
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.xlabel('Training episodes', fontsize=12)
# plt.ylabel('Reward', fontsize=12)
# plt.xlim(0, 13000)
# plt.ylim(0, 1)
# plt.legend(loc='lower right', fontsize=12)
# plt.show()

