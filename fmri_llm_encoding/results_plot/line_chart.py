import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    base_df = pd.read_csv('Base_final_results.csv')
    instruct_df = pd.read_csv('Instruct_final_results.csv')
except FileNotFoundError:
    print("can't find CSV")

run_cols = [col for col in base_df.columns if 'Run' in col]

def get_stats(df, cols):
    mean = df[cols].mean(axis=1)
    std = df[cols].std(axis=1)
    sem = std / np.sqrt(len(cols))
    return mean, sem

base_mean, base_sem = get_stats(base_df, run_cols)
instruct_mean, instruct_sem = get_stats(instruct_df, run_cols)
layers = base_df['Unnamed: 0']  

plt.figure(figsize=(12, 7)) 

plt.plot(layers, base_mean, label='Base Model (LLaMA-3.1-8B)', 
         color='#1f77b4', linewidth=2, marker='o', markersize=5)
plt.fill_between(layers, base_mean - base_sem, base_mean + base_sem, 
                 color='#1f77b4', alpha=0.2)
plt.plot(layers, instruct_mean, label='Instruct Model (LLaMA-3.1-8B-Instruct)', 
         color='#d62728', linewidth=2, marker='s', markersize=5)
plt.fill_between(layers, instruct_mean - instruct_sem, instruct_mean + instruct_sem, 
                 color='#d62728', alpha=0.2)

plt.xlabel('Transformer Layer', fontsize=12)
plt.ylabel('Encoding Performance (Pearson r)', fontsize=12)
plt.title('Brain Encoding: Base vs. Instruct Model across Layers', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='best')  
plt.grid(True, linestyle='--', alpha=0.6) 
plt.xlim(0, 32) 
plt.ylim(bottom=0) 

base_peak_idx = base_mean.idxmax()
instruct_peak_idx = instruct_mean.idxmax()
plt.axvline(x=base_peak_idx, color='#1f77b4', linestyle=':', alpha=0.5)
plt.axvline(x=instruct_peak_idx, color='#d62728', linestyle=':', alpha=0.5)

plt.tight_layout()

plt.savefig('encoding_comparison_plot.png', dpi=300)
print("图表已保存为 encoding_comparison_plot.png")

plt.show()