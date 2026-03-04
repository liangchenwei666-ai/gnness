import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from experiment_results.txt
data_noise = {
    'Noise': [0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1],
    'Classical': [72.10, 0.00, 0.00, 11.75, 0.00, 0.00],
    'Hybrid': [10.85, 11.20, 1.75, 0.00, 0.00, 0.00],
    'Meta (Neighbor)': [33.65, 29.10, 24.65, 11.75, 3.15, 0.00],
    'Meta (Top3)': [37.00, 32.05, 24.90, 11.70, 2.45, 0.00],
    'Meta (All)': [50.40, 35.50, 28.15, 18.75, 6.00, 0.00]
}

data_vsr = {
    'Noise': [0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1],
    'Classical': [100.00, 0.00, 0.00, 42.50, 0.05, 0.00],
    'Hybrid': [92.15, 89.65, 14.05, 0.00, 0.00, 0.00],
    'Meta (Neighbor)': [84.70, 85.70, 78.65, 19.40, 4.30, 0.00],
    'Meta (Top3)': [91.40, 91.65, 81.90, 18.40, 3.25, 0.00],
    'Meta (All)': [100.00, 100.00, 96.70, 53.40, 12.50, 0.00]
}

# Create Figure 4
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Accuracy (Main Axis)
# Using log scale for x-axis to visualize noise levels properly
# We replace 0 with a small value for log plotting or just use indices
noise_labels = ['$0$', '$10^{-6}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$']
x = np.arange(len(noise_labels))

ax1.plot(x, data_noise['Classical'], marker='o', label='Classical', linewidth=2, linestyle='-')
ax1.plot(x, data_noise['Hybrid'], marker='s', label='Hybrid (Original)', linewidth=2, linestyle='--')
ax1.plot(x, data_noise['Meta (Neighbor)'], marker='^', label='Meta (Neighbor)', linewidth=2, linestyle='-')
ax1.plot(x, data_noise['Meta (Top3)'], marker='v', label='Meta (Top-3)', linewidth=2, linestyle='-')
ax1.plot(x, data_noise['Meta (All)'], marker='*', label='Meta (All)', linewidth=2, linestyle='-', color='red')

ax1.set_xlabel('Noise level $\sigma$', fontsize=12)
ax1.set_ylabel('Rank Accuracy (%)', fontsize=12)
ax1.set_title('Noise Robustness at Degree $d=200$', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(noise_labels)
ax1.set_ylim(-5, 105)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper right', fontsize=10)

# Optional: Inset for VSR
# Create inset axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax_inset = inset_axes(ax1, width="40%", height="30%", loc='center right', borderpad=2)

ax_inset.plot(x, data_vsr['Classical'], marker='o', markersize=4, linestyle='-')
ax_inset.plot(x, data_vsr['Hybrid'], marker='s', markersize=4, linestyle='--')
ax_inset.plot(x, data_vsr['Meta (Neighbor)'], marker='^', markersize=4, linestyle='-')
ax_inset.plot(x, data_vsr['Meta (Top3)'], marker='v', markersize=4, linestyle='-')
ax_inset.plot(x, data_vsr['Meta (All)'], marker='*', markersize=4, linestyle='-', color='red')

ax_inset.set_xticks(x)
ax_inset.set_xticklabels(['0', '-6', '-4', '-3', '-2', '-1'], fontsize=8)
ax_inset.set_ylabel('VSR (%)', fontsize=9)
ax_inset.set_title('Verified Success Rate', fontsize=10)
ax_inset.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig('figure4_robustness.png', dpi=300)
plt.savefig('figure4_robustness.pdf')
print("Figure 4 saved as figure4_robustness.png and .pdf")
