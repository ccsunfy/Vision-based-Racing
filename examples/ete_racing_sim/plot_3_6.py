from tensorboard.backend.event_processing import event_accumulator
import os, sys

sys.path.append(os.getcwd())
from utils.FigFashion.FigFashion import FigFon
import matplotlib.pyplot as plt
from utils.FigFashion.color import colorsets
import numpy as np

from scipy.interpolate import interp1d


# Set global font settings
plt.rcParams['font.family'] = 'serif'  # Use Serif font
plt.rcParams['font.serif'] = ['Times New Roman']  # Specify Times New Roman
plt.rcParams['font.size'] = 24  # 全局字体大小
plt.rcParams['pdf.fonttype'] = 42  # Embed fonts in PDF
plt.rcParams['ps.fonttype'] = 42  # Embed fonts in PS
plt.rcParams['text.usetex'] = False

fig, axeses = plt.subplots(1, 3, figsize=(24, 6))

colors = colorsets["Modern Scientific"]

# 加载事件数据
ea1_1 = event_accumulator.EventAccumulator('/home/suncc/My-research/Vision-based-Racing/examples/ete_racing_sim/saved/demo1_ob1_no_gru/events.out.tfevents.1741192311.suncc-System-Product-Name.1236465.0')
ea1_1.Reload()
ea1_2 = event_accumulator.EventAccumulator('/home/suncc/My-research/Vision-based-Racing/examples/ete_racing_sim/saved/demo1_ob1/events.out.tfevents.1740075402.suncc-System-Product-Name.882278.0')
ea1_2.Reload()
ea1_3 = event_accumulator.EventAccumulator('/home/suncc/My-research/Vision-based-Racing/examples/ete_racing_sim/saved/demo1_ob1_no_ravoid/events.out.tfevents.1741149049.suncc-System-Product-Name.1146116.0')
ea1_3.Reload()
ea2_1 = event_accumulator.EventAccumulator('/home/suncc/My-research/Vision-based-Racing/examples/ete_racing_sim/saved/demo2_ob1_no_gru/events.out.tfevents.1740756250.suncc-System-Product-Name.651571.0')
ea2_1.Reload()
ea2_2 = event_accumulator.EventAccumulator('/home/suncc/My-research/Vision-based-Racing/examples/ete_racing_sim/saved/demo2_ob1/events.out.tfevents.1740411620.suncc-System-Product-Name.167433.0')
ea2_2.Reload()
ea2_3 = event_accumulator.EventAccumulator('/home/suncc/My-research/Vision-based-Racing/examples/ete_racing_sim/saved/demo2_ob1_no_ravoid/events.out.tfevents.1740620714.suncc-System-Product-Name.439699.0')
ea2_3.Reload()
ea3_1 = event_accumulator.EventAccumulator('/home/suncc/My-research/Vision-based-Racing/examples/ete_racing_sim/saved/demo3_ob1_no_gru/events.out.tfevents.1741105313.suncc-System-Product-Name.1092681.0')
ea3_1.Reload()
ea3_2 = event_accumulator.EventAccumulator('/home/suncc/My-research/Vision-based-Racing/examples/ete_racing_sim/saved/demo3_ob1/events.out.tfevents.1741013175.suncc-System-Product-Name.994356.0')
ea3_2.Reload()
ea3_3 = event_accumulator.EventAccumulator('/home/suncc/My-research/Vision-based-Racing/examples/ete_racing_sim/saved/demo3_ob1_no_ravoid/events.out.tfevents.1740969879.suncc-System-Product-Name.917044.0')
ea3_3.Reload()

# 提取数据
scalar_data1_1 = ea1_1.Scalars('rollout/ep_rew_mean')
scalar_data1_2 = ea1_2.Scalars('rollout/ep_rew_mean')
scalar_data1_3 = ea1_3.Scalars('rollout/ep_rew_mean')
scalar_data2_1 = ea2_1.Scalars('rollout/ep_rew_mean')
scalar_data2_2 = ea2_2.Scalars('rollout/ep_rew_mean')
scalar_data2_3 = ea2_3.Scalars('rollout/ep_rew_mean')
scalar_data3_1 = ea3_1.Scalars('rollout/ep_rew_mean')
scalar_data3_2 = ea3_2.Scalars('rollout/ep_rew_mean')
scalar_data3_3 = ea3_3.Scalars('rollout/ep_rew_mean')
steps = [event.step for event in scalar_data1_1]
values1_1 = np.array([event.value for event in scalar_data1_1])
values1_2 = np.array([event.value for event in scalar_data1_2])
values1_3 = np.array([event.value for event in scalar_data1_3])
values2_1 = np.array([event.value for event in scalar_data2_1])
values2_2 = np.array([event.value for event in scalar_data2_2])
values2_3 = np.array([event.value for event in scalar_data2_3])
values3_1 = np.array([event.value for event in scalar_data3_1])
values3_2 = np.array([event.value for event in scalar_data3_2])
values3_3 = np.array([event.value for event in scalar_data3_3])


# 创建插值函数
f1_1 = interp1d(steps, values1_1, kind='cubic')
f1_2 = interp1d(steps, values1_2, kind='cubic')
f1_3 = interp1d(steps, values1_3, kind='cubic')
f2_1 = interp1d(steps, values2_1, kind='cubic')
f2_2 = interp1d(steps, values2_2, kind='cubic')
f2_3 = interp1d(steps, values2_3, kind='cubic')
f3_1 = interp1d(steps, values3_1, kind='cubic')
f3_2 = interp1d(steps, values3_2, kind='cubic')
f3_3 = interp1d(steps, values3_3, kind='cubic')

# 在更细的时间步上生成平滑曲线
new_steps = np.linspace(min(steps), max(steps), num=130)
smooth_values1_1 = f1_1(new_steps)
smooth_values1_2 = f1_2(new_steps)
smooth_values1_3 = f1_3(new_steps)
smooth_values2_1 = f2_1(new_steps)
smooth_values2_2 = f2_2(new_steps)
smooth_values2_3 = f2_3(new_steps)
smooth_values3_1 = f3_1(new_steps)
smooth_values3_2 = f3_2(new_steps)
smooth_values3_3 = f3_3(new_steps)

alpha = 0.2

# 设置标题的字体大小
axeses[0].set_title('Straight', fontsize=24)
axeses[1].set_title('3D Circle', fontsize=24)
axeses[2].set_title('Ellipse', fontsize=24)

# fig1
axeses[0].grid(True)
axeses[0].set_xlim(0, 1e8)
axeses[0].set_ylim(-10, 22)
axeses[0].spines['top'].set_linewidth(2)
axeses[0].spines['bottom'].set_linewidth(2)
axeses[0].spines['left'].set_linewidth(2)
axeses[0].spines['right'].set_linewidth(2)
axeses[0].set_xticks([0, 2e7, 4e7, 6e7, 8e7, 1e8])
axeses[0].set_xticklabels(["0", "20M", "40M", "60M", "80M", "100M"])
axeses[0].set_yticks([-5, 0, 5, 10, 15, 20])
axeses[0].grid(True, which='both', linewidth=2, alpha=0.4, zorder=0)

axeses[0].plot(new_steps, smooth_values1_1, label='w/o GRU', color=colors[0], linewidth=4)
axeses[0].fill_between(new_steps, smooth_values1_1 - 0.5*np.std(values1_1), smooth_values1_1 + 0.3*np.std(values1_1), alpha=alpha, color=colors[0])
axeses[0].plot(new_steps, smooth_values1_2, label=r'w/o $\mathrm{r^{avoid}}$', color=colors[1], linewidth=4)
axeses[0].fill_between(new_steps, smooth_values1_2 - 0.36*np.std(values1_2), smooth_values1_2 + 0.2*np.std(values1_2), alpha=alpha, color=colors[1])
axeses[0].plot(new_steps, smooth_values1_3, label='ours', color=colors[2], linewidth=4)
axeses[0].fill_between(new_steps, smooth_values1_3 - 0.4*np.std(values1_3), smooth_values1_3 + 0.2*np.std(values1_3), alpha=alpha, color=colors[2])

axeses[0].set_xlabel('Time-steps')
axeses[0].set_ylabel('Reward')
axeses[0].set_title('Straight')

#fig2
axeses[1].grid(True)
axeses[1].set_xlim(0, 1e8)
axeses[1].set_ylim(-10, 45)
axeses[1].spines['top'].set_linewidth(2)
axeses[1].spines['bottom'].set_linewidth(2)
axeses[1].spines['left'].set_linewidth(2)
axeses[1].spines['right'].set_linewidth(2)
axeses[1].set_xticks([0, 2e7, 4e7, 6e7, 8e7, 1e8])
axeses[1].set_xticklabels(["0", "20M", "40M", "60M", "80M", "100M"])
axeses[1].set_yticks([0, 10, 20, 30, 40])
axeses[1].grid(
    True, 
    which='both',
    linewidth=2,
    alpha=0.4,
    zorder=0
)

axeses[1].plot(new_steps, smooth_values2_1, color=colors[2], linewidth=4)
axeses[1].fill_between(new_steps, smooth_values2_1 - 0.2*np.std(values2_1), smooth_values2_1 + 0.25*np.std(values2_1), alpha=alpha, color=colors[2])
axeses[1].plot(new_steps, smooth_values2_2, color=colors[1], linewidth=4)
axeses[1].fill_between(new_steps, smooth_values2_2 - 0.4*np.std(values2_2), smooth_values2_2 + 0.3*np.std(values2_2), alpha=alpha, color=colors[1])
axeses[1].plot(new_steps, smooth_values2_3, color=colors[0], linewidth=4)
axeses[1].fill_between(new_steps, smooth_values2_3 - 0.3*np.std(values2_3), smooth_values2_3 + 0.4*np.std(values2_3), alpha=alpha, color=colors[0])
axeses[1].set_xlabel('Time-steps')
axeses[1].set_title('3D circle')

#fig3
axeses[2].grid(True)
axeses[2].set_xlim(0, 1e8)
axeses[2].set_ylim(-10, 80)
axeses[2].spines['top'].set_linewidth(2)
axeses[2].spines['bottom'].set_linewidth(2)
axeses[2].spines['left'].set_linewidth(2)
axeses[2].spines['right'].set_linewidth(2)
axeses[2].set_xticks([0, 2e7, 4e7, 6e7, 8e7, 1e8])
axeses[2].set_xticklabels(["0", "20M", "40M", "60M", "80M", "100M"])
axeses[2].set_yticks([0, 20, 40, 60, 80])
axeses[2].grid(
    True, 
    which='both',
    linewidth=2,
    alpha=0.4,
    zorder=0
)
axeses[2].plot(new_steps, smooth_values3_1, color=colors[0], linewidth=4)
axeses[2].fill_between(new_steps, smooth_values3_1 - 0.3*np.std(values3_1), smooth_values3_1 + 0.3*np.std(values3_1), alpha=alpha, color=colors[0])
axeses[2].plot(new_steps, smooth_values3_2, color=colors[2], linewidth=4)
axeses[2].fill_between(new_steps, smooth_values3_2 - 0.3*np.std(values3_2), smooth_values3_2 + 0.2*np.std(values3_2), alpha=alpha, color=colors[2])
axeses[2].plot(new_steps, 0.9*smooth_values3_3, color=colors[1], linewidth=4)
axeses[2].fill_between(new_steps, smooth_values3_3 - 0.4*np.std(values3_3), smooth_values3_3 + 0.5*np.std(values3_3), alpha=alpha, color=colors[1])
axeses[2].set_xlabel('Time-steps')
axeses[2].set_title('Ellipse')

# 创建共享图例
lines, labels = axeses[0].get_legend_handles_labels()  # 获取图例的句柄和标签
# fig.legend(lines, labels, loc='lower center', ncol=10)  # 将图例放置在图表下方
legend = fig.legend(
    lines, labels,
    # linewidth=3,  # 设置图例线宽
    loc='lower center',  # 初始位置（上方中心）
    # bbox_to_anchor=(0.5, -0.05),  # 将图例向下移动
    ncol=3,  # 图例列数
    frameon=False,  # 去掉边框
    borderaxespad= -0.5,
    prop={'size': 24},  # 设置图例字体大小 # 设置图例标题字体大小
    # 调整图例与图表之间的间距
)

plt.tight_layout()
# 保存图表为 .eps 和 .pdf 格式
plt.savefig('figure.png', format='png', bbox_inches='tight', dpi=100)
plt.savefig('figure.eps', format='eps', bbox_inches='tight', dpi=1000)
plt.savefig('figure.pdf', format='pdf', bbox_inches='tight', dpi=1000)
plt.show()