import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Load benchmark results
df = pd.read_csv('benchmark_results.csv')
df = df.dropna(subset=['Time (s)', 'Peak Memory (MB)'])

def plot_time(df):
    fig, ax = plt.subplots(figsize=(12, 6))

    model_styles = {
        'Transolver': '--',
        'HAETransolver': '-'
    }

    unique_slices = sorted(df['Slices'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_slices)))
    color_map = dict(zip(unique_slices, colors))

    for model in df['Model'].unique():
        for slice_val in unique_slices:
            subset = df[(df['Model'] == model) & (df['Slices'] == slice_val)].dropna()
            if not subset.empty:
                ax.plot(subset['Points'], subset['Time (s)'],
                        linestyle=model_styles[model],
                        color=color_map[slice_val],
                        marker='o')

    ax.set_title('Benchmark Time vs Number of Points\n(HAETransolver = solid, Transolver = dashed)')
    ax.set_xlabel('Number of Points')
    ax.set_ylabel('Time (s)')
    ax.grid(True)

    # X-axis ticks
    unique_points = sorted(df['Points'].unique())
    ax.set_xticks(unique_points)
    ax.set_xticklabels([f"{int(p):,}" for p in unique_points], rotation=45)

    # Legend: only slices
    slice_handles = [mlines.Line2D([], [], color=color_map[s], marker='o', linestyle='-', label=f"Slices = {s}") for s in unique_slices]
    ax.legend(slice_handles, [f"Slices = {s}" for s in unique_slices], fontsize=9, title="Color Legend", ncol=3)

    plt.tight_layout()
    plt.savefig('benchmark_time.png')

def plot_memory(df):
    fig, ax = plt.subplots(figsize=(12, 6))

    model_styles = {
        'Transolver': '--',
        'HAETransolver': '-'
    }

    unique_slices = sorted(df['Slices'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_slices)))
    color_map = dict(zip(unique_slices, colors))

    for model in df['Model'].unique():
        for slice_val in unique_slices:
            subset = df[(df['Model'] == model) & (df['Slices'] == slice_val)].dropna()
            if not subset.empty:
                ax.plot(subset['Points'], subset['Peak Memory (MB)'],
                        linestyle=model_styles[model],
                        color=color_map[slice_val],
                        marker='o')

    # Memory limit lines
    a100_line = ax.axhline(y=40960, color='red', linestyle=':', linewidth=1.5, label='A100 40GB Limit')
    h100_line = ax.axhline(y=94000, color='blue', linestyle=':', linewidth=1.5, label='H100 94GB Limit')

    ax.set_title('Peak Memory Usage vs Number of Points\n(HAETransolver = solid, Transolver = dashed)')
    ax.set_xlabel('Number of Points')
    ax.set_ylabel('Peak Memory (MB)')
    ax.grid(True)

    # X-axis ticks
    unique_points = sorted(df['Points'].unique())
    ax.set_xticks(unique_points)
    ax.set_xticklabels([f"{int(p):,}" for p in unique_points], rotation=45)

    # Custom legend: slice colors + memory limits
    slice_handles = [mlines.Line2D([], [], color=color_map[s], marker='o', linestyle='-', label=f"Slices = {s}") for s in unique_slices]
    handles = slice_handles + [a100_line, h100_line]
    labels = [f"Slices = {s}" for s in unique_slices] + ['A100 80GB Limit', 'H100 94GB Limit']
    ax.legend(handles, labels, fontsize=9, title="Legend", ncol=3)

    plt.tight_layout()
    plt.savefig('benchmark_memory.png')

# Run both plots
plot_time(df)
plot_memory(df)