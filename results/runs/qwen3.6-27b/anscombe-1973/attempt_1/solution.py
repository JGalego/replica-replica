import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Anscombe's quartet - canonical data values
datasets = {
    'I': {
        'x': np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]),
        'y': np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
    },
    'II': {
        'x': np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]),
        'y': np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
    },
    'III': {
        'x': np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]),
        'y': np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
    },
    'IV': {
        'x': np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19]),
        'y': np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50])
    }
}

# Compute regression line y = 3 + 0.5x
x_line = np.linspace(3, 20, 100)
y_line = 3.0 + 0.5 * x_line

# Create figure with 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

labels = ['I', 'II', 'III', 'IV']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for idx, label in enumerate(labels):
    row, col = positions[idx]
    ax = axes[row, col]
    data = datasets[label]
    
    # Plot scatter points
    ax.scatter(data['x'], data['y'], color='black', s=50, zorder=5, edgecolors='none')
    
    # Plot regression line
    ax.plot(x_line, y_line, color='red', linewidth=1.5, linestyle='-')
    
    # Set axis limits
    ax.set_xlim(3, 20)
    ax.set_ylim(3, 13)
    
    # Set title
    ax.set_title(f'Dataset {label}', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set ticks
    ax.set_xticks(np.arange(4, 21, 2))
    ax.set_yticks(np.arange(4, 14, 2))

# Set shared axis labels
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 1].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('y', fontsize=12)
axes[1, 0].set_ylabel('y', fontsize=12)

# Add common summary statistics as text annotation
summary_text = (
    r'$\bar{x}=9.0,\ \bar{y}=7.50$' + '\n' +
    r'$s_x^2=11.0,\ s_y^2\approx4.13$' + '\n' +
    r'$r=0.816,\ y=3.00+0.500x$'
)
fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))

plt.tight_layout(rect=[0, 0.08, 1, 1])

from replica_replica.harness import submit
submit(fig)