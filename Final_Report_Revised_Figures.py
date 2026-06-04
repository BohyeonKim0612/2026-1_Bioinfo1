import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import urllib.request
import ssl
import matplotlib.font_manager as fm

# Load custom Pretendard font
font_paths = ['./Pretendard-SemiBold.ttf', './Pretendard-Bold.ttf']
for fp in font_paths:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)

# Global styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
sns.set_context('paper', font_scale=1.4)
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['font.weight'] = '600' # SemiBold
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['grid.linestyle'] = '--'

out_dir = 'report_figures_revised'
os.makedirs(out_dir, exist_ok=True)

# 1. Data Loading (Combined)
filepath = 'binfo1-work/read-counts.txt'
if not os.path.exists(filepath):
    filepath = 'read-counts.txt'

counts = pd.read_csv(filepath, sep='\t', comment='#', index_col=0)

min_count = 10
mask = (
    (counts['RNA-control.bam'] >= min_count) &
    (counts['RNA-siLin28a.bam'] >= min_count) &
    (counts['RNA-siLuc.bam'] >= min_count) &
    (counts['RPF-siLin28a.bam'] >= min_count) &
    (counts['RPF-siLuc.bam'] >= min_count)
)
filtered = counts[mask].copy()

filtered['TE_siLuc']    = filtered['RPF-siLuc.bam']    / filtered['RNA-siLuc.bam']
filtered['TE_siLin28a'] = filtered['RPF-siLin28a.bam'] / filtered['RNA-siLin28a.bam']
filtered['log2_fc_TE']  = np.log2(filtered['TE_siLin28a'] / filtered['TE_siLuc'])

filtered['CLIP_enrich'] = filtered['CLIP-35L33G.bam'] / filtered['RNA-control.bam']
filtered.loc[filtered['CLIP-35L33G.bam'] == 0, 'CLIP_enrich'] = 0

q01 = filtered['log2_fc_TE'].quantile(0.01)
q99 = filtered['log2_fc_TE'].quantile(0.99)
filtered = filtered[(filtered['log2_fc_TE'] >= q01) & (filtered['log2_fc_TE'] <= q99)].copy()
filtered['gene_id_base'] = filtered.index.str.split('.').str[0]

# Definitions
clip_pos = filtered[filtered['CLIP_enrich'] > 0]
threshold = clip_pos['CLIP_enrich'].quantile(0.8)
targets = filtered[filtered['CLIP_enrich'] >= threshold]
non_targets = filtered[filtered['CLIP_enrich'] == 0]

# --- Figure 1: CDF ---
def get_cdf_coords(data):
    sorted_data = np.sort(data)
    y_coords = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, y_coords

x_tgt, y_tgt = get_cdf_coords(targets['log2_fc_TE'])
x_ntgt, y_ntgt = get_cdf_coords(non_targets['log2_fc_TE'])
ks_stat, p_val = stats.ks_2samp(targets['log2_fc_TE'], non_targets['log2_fc_TE'])

fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=300)
ax1.plot(x_ntgt, y_ntgt, label='Non-Targets', color='grey', linewidth=2.5, linestyle='--')
ax1.plot(x_tgt, y_tgt, label='Lin28a Targets', color='#e74c3c', linewidth=3, alpha=0.9)
ax1.axvline(x=0, color='#2c3e50', linestyle='-', linewidth=1, alpha=0.5)

ax1.set_title('Translation Efficiency Changes', pad=15)
ax1.set_xlabel(r'$\log_2$ Fold Change of TE')
ax1.set_ylabel('Cumulative Fraction')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(0, 1.05)

info_text = f'KS-Test P-value: {p_val:.2e}'
ax1.text(0.05, 0.90, info_text, transform=ax1.transAxes, fontsize=11, 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ddd', boxstyle='round,pad=0.5'))
ax1.legend(loc='lower right', frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig(f'{out_dir}/Revised_Fig1_CDF.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Figure 2: Violin + Swarmplot (Quartiles) ---
clip_positive = filtered[filtered['CLIP_enrich'] > 0].copy()
clip_positive['Quartile'] = pd.qcut(clip_positive['CLIP_enrich'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

non_target_df = non_targets.copy()
non_target_df['Quartile'] = 'Non-target'

analysis2_data = pd.concat([non_target_df, clip_positive])
order = ['Non-target', 'Q1', 'Q2', 'Q3', 'Q4']

fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=300)
# Reduced width for violin
sns.violinplot(x='Quartile', y='log2_fc_TE', data=analysis2_data, order=order, 
               inner=None, linewidth=1.2, ax=ax2, alpha=0.8, width=0.6, palette='husl')
# Add median and quartiles manually or use boxplot inside
sns.boxplot(x='Quartile', y='log2_fc_TE', data=analysis2_data, order=order, 
            width=0.2, boxprops={'facecolor':'none', 'zorder':10}, showcaps=False, 
            whiskerprops={'linewidth':2}, medianprops={'linewidth':2, 'color':'#333'}, 
            showfliers=False, ax=ax2)

ax2.axhline(0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.6)
ax2.set_title('TE Change by Binding Strength', pad=15)
ax2.set_xlabel('Lin28a Binding Strength (Quartiles)')
ax2.set_ylabel(r'$\log_2$ Fold Change of TE')

# Add asterisks for significance (vs Non-target)
control_data = analysis2_data[analysis2_data['Quartile'] == 'Non-target']['log2_fc_TE']
y_max = analysis2_data['log2_fc_TE'].max()
y_offset = (y_max - analysis2_data['log2_fc_TE'].min()) * 0.05
for i, group in enumerate(order[1:], 1):
    group_data = analysis2_data[analysis2_data['Quartile'] == group]['log2_fc_TE']
    stat, p = stats.mannwhitneyu(control_data, group_data, alternative='two-sided')
    if p < 0.0001: sig = '****'
    elif p < 0.001: sig = '***'
    elif p < 0.01: sig = '**'
    elif p < 0.05: sig = '*'
    else: sig = 'ns'
    
    if sig != 'ns':
        ax2.text(i, y_max + y_offset, sig, ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')

ax2.set_ylim(analysis2_data['log2_fc_TE'].min() - y_offset*2, y_max + y_offset * 4)

sns.despine()
plt.tight_layout()
plt.savefig(f'{out_dir}/Revised_Fig2_Violin_Quartiles.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Figure 3: Scatter ---
plot_data = targets.copy()
plot_data['log2_CLIP'] = np.log2(plot_data['CLIP_enrich'])
spearman_corr, p_value_3 = stats.spearmanr(plot_data['CLIP_enrich'], plot_data['log2_fc_TE'])

fig3, ax3 = plt.subplots(figsize=(6, 5), dpi=300)
ax3.scatter(plot_data['log2_CLIP'], plot_data['log2_fc_TE'], alpha=0.4, s=25, edgecolors='none', color='gray')
sns.regplot(x='log2_CLIP', y='log2_fc_TE', data=plot_data, scatter=False, line_kws={'linewidth':2}, ax=ax3)

ax3.set_title('CLIP Enrichment vs TE Change', pad=15)
ax3.set_xlabel(r'$\log_2$(CLIP Enrichment)')
ax3.set_ylabel(r'$\log_2$ Fold Change of TE')

info_text_3 = f'Spearman r = {spearman_corr:.3f}\nP-value = {p_value_3:.2e}'
ax3.text(0.05, 0.95, info_text_3, transform=ax3.transAxes, fontsize=11, verticalalignment='top', 
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ddd', boxstyle='round,pad=0.5'))

sns.despine()
plt.tight_layout()
plt.savefig(f'{out_dir}/Revised_Fig3_Scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Figure 4: Localization ---
ssl_ctx = ssl._create_unverified_context()
url = 'https://hyeshik.qbio.io/binfo/mouselocalization-20210507.txt'
with urllib.request.urlopen(url, context=ssl_ctx) as resp:
    mouselocal = pd.read_csv(resp, sep='\t')

loc_idx = mouselocal.set_index('gene_id')[['type']]
merged = filtered.merge(loc_idx, left_on='gene_id_base', right_index=True, how='inner')

def classify(row):
    if row['CLIP_enrich'] >= threshold: return 'Target'
    elif row['CLIP_enrich'] == 0: return 'Non-Target'
    return 'Intermediate'
    
merged['Target_Group'] = merged.apply(classify, axis=1)
target_merged = merged[merged['Target_Group'] == 'Target'].copy()

type_counts = target_merged['type'].value_counts()
valid_types = type_counts[type_counts >= 15].index.tolist()
plot_data_loc = target_merged[target_merged['type'].isin(valid_types)].copy()

order_loc = plot_data_loc.groupby('type')['log2_fc_TE'].median().sort_values(ascending=False).index.tolist()

fig4, ax4 = plt.subplots(figsize=(8, 5), dpi=300)
sns.violinplot(x='type', y='log2_fc_TE', data=plot_data_loc, 
               inner=None, order=order_loc, ax=ax4, linewidth=1.2, alpha=0.8, width=0.6, palette='Set3')
sns.boxplot(x='type', y='log2_fc_TE', data=plot_data_loc, order=order_loc, 
            width=0.2, boxprops={'facecolor':'none', 'zorder':10}, showcaps=False, 
            whiskerprops={'linewidth':2}, medianprops={'linewidth':2, 'color':'#333'}, 
            showfliers=False, ax=ax4)

ax4.axhline(0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.6)
ax4.set_title('TE Change by Subcellular Localization', pad=15)
ax4.set_xlabel('Subcellular Localization')
ax4.set_ylabel(r'$\log_2$ Fold Change of TE')
ax4.tick_params(axis='x', rotation=15)

c_names = [t for t in order_loc if t.lower() == 'cytoplasm']
control_group = c_names[0] if c_names else order_loc[0]
control_data = plot_data_loc[plot_data_loc['type'] == control_group]['log2_fc_TE']

y_max = plot_data_loc['log2_fc_TE'].max()
y_offset = (y_max - plot_data_loc['log2_fc_TE'].min()) * 0.05

for i, group in enumerate(order_loc):
    if group == control_group:
        continue
    group_data = plot_data_loc[plot_data_loc['type'] == group]['log2_fc_TE']
    stat, p = stats.mannwhitneyu(control_data, group_data, alternative='two-sided')
    if p < 0.0001: sig = '****'
    elif p < 0.001: sig = '***'
    elif p < 0.01: sig = '**'
    elif p < 0.05: sig = '*'
    else: sig = 'ns'
    
    if sig != 'ns':
        ax4.text(i, y_max + y_offset, sig, ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')

ax4.set_ylim(plot_data_loc['log2_fc_TE'].min() - y_offset*2, y_max + y_offset * 4)

sns.despine()
plt.tight_layout()
plt.savefig(f'{out_dir}/Revised_Fig4_Violin_Loc.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Figure 5: CLIP Enrichment by Localization ---
valid_types_all = merged['type'].value_counts()[merged['type'].value_counts() >= 15].index.tolist()
plot_data_all = merged[merged['type'].isin(valid_types_all)].copy()

plot_data_clip = plot_data_all[plot_data_all['CLIP_enrich'] > 0].copy()
plot_data_clip['log2_CLIP'] = np.log2(plot_data_clip['CLIP_enrich'])

order_loc_all = plot_data_clip.groupby('type')['log2_CLIP'].median().sort_values(ascending=False).index.tolist()

fig5, ax5 = plt.subplots(figsize=(8, 5), dpi=300)
sns.violinplot(x='type', y='log2_CLIP', data=plot_data_clip, 
               inner=None, order=order_loc_all, ax=ax5, linewidth=1.2, alpha=0.8, width=0.6, palette='Set3')
sns.boxplot(x='type', y='log2_CLIP', data=plot_data_clip, order=order_loc_all, 
            width=0.2, boxprops={'facecolor':'none', 'zorder':10}, showcaps=False, 
            whiskerprops={'linewidth':2}, medianprops={'linewidth':2, 'color':'#333'}, 
            showfliers=False, ax=ax5)

ax5.set_title('LIN28A Binding Strength (CLIP) by Localization', pad=15)
ax5.set_xlabel('Subcellular Localization')
ax5.set_ylabel(r'$\log_2$(CLIP Enrichment)')
ax5.tick_params(axis='x', rotation=15)

c_names_all = [t for t in order_loc_all if t.lower() == 'cytoplasm']
control_group_all = c_names_all[0] if c_names_all else order_loc_all[0]
control_data_all = plot_data_clip[plot_data_clip['type'] == control_group_all]['log2_CLIP']

y_max_5 = plot_data_clip['log2_CLIP'].max()
y_offset_5 = (y_max_5 - plot_data_clip['log2_CLIP'].min()) * 0.05

for i, group in enumerate(order_loc_all):
    if group == control_group_all:
        continue
    group_data_all = plot_data_clip[plot_data_clip['type'] == group]['log2_CLIP']
    stat, p = stats.mannwhitneyu(control_data_all, group_data_all, alternative='two-sided')
    if p < 0.0001: sig = '****'
    elif p < 0.001: sig = '***'
    elif p < 0.01: sig = '**'
    elif p < 0.05: sig = '*'
    else: sig = 'ns'
    
    if sig != 'ns':
        ax5.text(i, y_max_5 + y_offset_5, sig, ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')

ax5.set_ylim(plot_data_clip['log2_CLIP'].min() - y_offset_5*2, y_max_5 + y_offset_5 * 4)

sns.despine()
plt.tight_layout()
plt.savefig(f'{out_dir}/Revised_Fig5_Violin_CLIP_Loc.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figures successfully generated in report_figures_revised/")
