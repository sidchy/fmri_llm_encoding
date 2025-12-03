import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    base_df = pd.read_csv('Base_final_results.csv', index_col=0) 
    instruct_df = pd.read_csv('Instruct_final_results.csv', index_col=0)
except FileNotFoundError:
    print("can't fine CSV")
    runs = [f'Run{i}' for i in range(15, 24)]
    base_df = pd.DataFrame(np.random.rand(33, 9) * 0.4, columns=runs)
    instruct_df = base_df + np.random.normal(0, 0.02, (33, 9))

run_cols = [col for col in base_df.columns if 'Run' in col]

base_mean = base_df[run_cols].mean(axis=1)
instruct_mean = instruct_df[run_cols].mean(axis=1)
layers = base_df.index

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 图表 A: 差异图 (±0.1)
# ==========================================
def plot_difference():
    diff = instruct_mean - base_mean
    
    plt.figure(figsize=(12, 6))
    plt.plot(layers, diff, color='black', linewidth=1.5, label='Difference (Instruct - Base)')
    
    plt.fill_between(layers, diff, 0, where=(diff > 0), color='#d62728', alpha=0.3, label='Improvement')
    plt.fill_between(layers, diff, 0, where=(diff <= 0), color='#1f77b4', alpha=0.3, label='Decline')
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    plt.ylim(-0.1, 0.1) 

    plt.xlabel('Transformer Layer', fontsize=12)
    plt.ylabel('Δ Pearson r (Instruct - Base)', fontsize=12)
    plt.title('A. Impact of Instruction Tuning across Layers', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('plot_A_difference.png', dpi=300)
    plt.show()

# ==========================================
# 图表 B: 热力图
# ==========================================
def plot_heatmap():
    plt.figure(figsize=(14, 6))
    vmin = base_df[run_cols].min().min()
    vmax = base_df[run_cols].max().max()
    
    sns.heatmap(base_df[run_cols].T, cmap="viridis", vmin=vmin, vmax=vmax, 
                cbar_kws={'label': 'Pearson r'})
    
    plt.xlabel('Transformer Layer', fontsize=12)
    plt.ylabel('Subject / Run ID', fontsize=12)
    plt.title('B. Encoding Performance Stability (Base Model)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot_B_heatmap.png', dpi=300)
    plt.show()

# ==========================================
# 图表 C: 散点相关图
# ==========================================
def plot_scatter():
    plt.figure(figsize=(8, 8))
    
    sc = plt.scatter(base_mean, instruct_mean, c=layers, cmap='Blues', s=100, edgecolors='k', alpha=0.8)
    
    all_vals = pd.concat([base_mean, instruct_mean])
    min_val = all_vals.min() * 0.95
    max_val = all_vals.max() * 1.05
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (No Difference)')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.colorbar(sc, label='Layer Depth')
    plt.xlabel('Base Model Performance (r)', fontsize=12)
    plt.ylabel('Instruct Model Performance (r)', fontsize=12)
    plt.title('C. Layer-wise Performance Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot_C_scatter.png', dpi=300)
    plt.show()

# ==========================================
# 图表 D: 箱线图
# ==========================================
def plot_boxplot():
    base_melt = base_df[run_cols].reset_index().melt(id_vars='index', var_name='Run', value_name='r')
    base_melt['Model'] = 'Base'
    instruct_melt = instruct_df[run_cols].reset_index().melt(id_vars='index', var_name='Run', value_name='r')
    instruct_melt['Model'] = 'Instruct'
    
    combined_df = pd.concat([base_melt, instruct_melt])
    combined_df.rename(columns={'index': 'Layer'}, inplace=True)

    plt.figure(figsize=(16, 7))
    filter_mask = combined_df['Layer'] % 2 == 0 
    sns.boxplot(x='Layer', y='r', hue='Model', data=combined_df[filter_mask], 
                palette={'Base': '#1f77b4', 'Instruct': '#d62728'}, linewidth=1.2, fliersize=3)
    
    plt.xlabel('Transformer Layer (Even numbers)', fontsize=12)
    plt.ylabel('Encoding Performance (r)', fontsize=12)
    plt.title('D. Distribution of Performance across Runs', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plot_D_boxplot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    print("图表 A ")
    plot_difference()
    
    print("图表 B...")
    plot_heatmap()
    
    print("图表 C")
    plot_scatter()
    
    print("图表 D")
    plot_boxplot()
    
    print("Done")