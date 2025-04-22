import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from sklearn.preprocessing import MinMaxScaler

# Read the data (make sure the CSV file is in the same directory as this script)
df = pd.read_csv("C:/Users/Lenovo/Desktop/rm/cw2/data/Results_21MAR2022_nokcaladjust.csv")


# ========================
# Data cleansing and pre-processing
# ========================

# Data cleansing and pre-processing retains only records with sufficient sample size (>= 50 persons)
df = df[df['n_participants'] >= 50]

# Deleting records with missing values
df = df.dropna()

# Selection of the type of diet analysed
selected_diets = ['fish', 'meat', 'meat100', 'vegetarian']
df = df[df['diet_group'].isin(selected_diets)]

# Select the type of diet analysed to create a ‘Diet Type + Gender’ combination label.
df['group'] = df['diet_group'] + '_' + df['sex']

# Environmental impact indicators to be analysed
metrics = [
    'mean_ghgs', 'mean_land', 'mean_watscar',
    'mean_eut', 'mean_ghgs_ch4', 'mean_ghgs_n2o',
    'mean_bio', 'mean_watuse', 'mean_acid'
]

# Group by group and calculate the mean
grouped = df.groupby('group')[metrics].mean().reset_index()

# Normalisation of all indicators (0 to 1)）
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(grouped[metrics])
normalized_df = pd.DataFrame(normalized_data, columns=metrics)
normalized_df.insert(0, 'group', grouped['group'])

# ========================
# Radar mapping
# ========================

# Radar chart structural parameters
categories = metrics
num_vars = len(categories)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # 闭合


plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Plotting the graph of each group
for i, row in normalized_df.iterrows():
    values = row[categories].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=row['group'])
    ax.fill(angles, values, alpha=0.1)

# Setting the coordinate labels
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
