import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from sklearn.preprocessing import MinMaxScaler

# 读取数据（确保 CSV 文件和此脚本在同一目录）
df = pd.read_csv("C:/Users/Lenovo/Desktop/rm/cw2/data/Results_21MAR2022_nokcaladjust.csv")


# ========================
# 数据清洗与预处理
# ========================

# 只保留样本数量足够的记录（>= 50人）
df = df[df['n_participants'] >= 50]

# 删除有缺失值的记录
df = df.dropna()

# 选择分析的饮食类型
selected_diets = ['fish', 'meat', 'meat100', 'vegetarian']
df = df[df['diet_group'].isin(selected_diets)]

# 创建“饮食类型 + 性别”的组合标签
df['group'] = df['diet_group'] + '_' + df['sex']

# 需要分析的环境影响指标
metrics = [
    'mean_ghgs', 'mean_land', 'mean_watscar',
    'mean_eut', 'mean_ghgs_ch4', 'mean_ghgs_n2o',
    'mean_bio', 'mean_watuse', 'mean_acid'
]

# 按 group 分组并计算平均值
grouped = df.groupby('group')[metrics].mean().reset_index()

# 对所有指标做归一化（0~1）
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(grouped[metrics])
normalized_df = pd.DataFrame(normalized_data, columns=metrics)
normalized_df.insert(0, 'group', grouped['group'])

# ========================
# 雷达图绘制
# ========================

# 雷达图结构参数
categories = metrics
num_vars = len(categories)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # 闭合

# 创建图表
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# 绘制每个 group 的图形
for i, row in normalized_df.iterrows():
    values = row[categories].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=row['group'])
    ax.fill(angles, values, alpha=0.1)

# 设置坐标标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
plt.title("Radar Chart: Environmental Impact by Diet & Gender (Normalized & Cleaned)", y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()

import os
os.makedirs("figure", exist_ok=True)
plt.savefig("figure/radar_chart_diet_gender_cleaned.png", dpi=300)

plt.show()
