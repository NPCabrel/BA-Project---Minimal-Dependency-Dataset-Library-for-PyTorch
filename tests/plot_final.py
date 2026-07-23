#!/usr/bin/env python3
"""Final comparison plot: local vs remote storage."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Local SSD\n(16 workers)', 'Remote BeeGFS\n(8 workers)']
ours = [1787, 1291]
pytorch = [2831, 1661]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width/2, ours, width, label='Our DataLoader', color='#2196F3')
bars2 = ax.bar(x + width/2, pytorch, width, label='PyTorch DataLoader', color='#4CAF50')

# Add ratio labels
for i, (o, p) in enumerate(zip(ours, pytorch)):
    ratio = p / o
    ax.text(x[i], max(o, p) + 50, f'{ratio:.2f}x', ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('Throughput (samples/s)', fontsize=12)
ax.set_title('DataLoader Performance: Local vs Remote Storage', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(int(bar.get_height())), ha='center', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(int(bar.get_height())), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('/netscratch/nague/plot_local_vs_remote.png', dpi=150)
print("Saved: plot_local_vs_remote.png")

# Copy to docs
import os
os.system('cp /netscratch/nague/plot_local_vs_remote.png ~/bachelor-project/docs/images/')
print("Copied to docs/images/")
