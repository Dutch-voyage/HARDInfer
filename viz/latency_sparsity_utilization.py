import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
file_format = "png"
scale = 1.5
plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24
})
prepare_color = '#94becf'
compute_color = '#f9bebe'

def load_sparse_latency_data():
    """Load the sparse pattern to latency data"""
    data = np.load("sparse_pattern_to_latency.npz", allow_pickle=True).item()
    return data["HA_flatten_max_prefill"]

def extract_latency_data(config_data):
    """Extract prepare and compute latency values"""
    start_vals = []
    sparse_ratios = []
    prepare_latencies = []
    total_compute_latencies = []

    for entry in config_data:
        start_vals.append(entry['stats_val'])
        sparse_ratios.append(entry['sparse_ratio'])

        if entry['stats']:
            prepare_latencies.append(entry['stats'][0]['prepare_mean_ms'])
            total_compute_latencies.append(entry['stats'][0]['compute_mean_ms'] * 36)  # 36 layers
        else:
            prepare_latencies.append(0)
            total_compute_latencies.append(0)

    # Create unique sorted lists
    unique_starts = sorted(set(start_vals))
    unique_ratios = sorted(set(sparse_ratios))

    # Create matrices (transposed: rows=sparse_ratios, cols=start_vals)
    prepare_matrix = np.zeros((len(unique_ratios), len(unique_starts)))
    total_compute_matrix = np.zeros((len(unique_ratios), len(unique_starts)))

    # Fill matrices (transposed: i=ratio index, j=start index)
    for start_val, sparse_ratio, prep_latency, total_comp_latency in zip(
        start_vals, sparse_ratios, prepare_latencies, total_compute_latencies):

        i = unique_ratios.index(sparse_ratio)
        j = unique_starts.index(start_val)
        prepare_matrix[i, j] = prep_latency
        total_compute_matrix[i, j] = total_comp_latency

    return prepare_matrix, total_compute_matrix, unique_ratios, unique_starts

def create_latency_breakdown():
    """Create breakdown visualization showing prepare and total compute latency merged in same figure"""

    # Load data
    config_data = load_sparse_latency_data()

    # Extract latency matrices (now: sparse_ratios on y-axis, start_vals on x-axis)
    prepare_matrix, total_compute_matrix, sparse_ratios, start_vals = extract_latency_data(config_data)

    print(f"Matrix shapes: {prepare_matrix.shape}")
    print(f"Prepare latency range: {np.min(prepare_matrix):.4f} - {np.max(prepare_matrix):.4f} ms")
    print(f"Total compute latency range: {np.min(total_compute_matrix):.4f} - {np.max(total_compute_matrix):.4f} ms")

    H = len(start_vals)  # x-axis (columns)
    W = len(sparse_ratios)  # y-axis (rows)

    # Count valid entries (upper triangle where row >= col, i.e., sparse_ratio >= start_val)
    valid_count = np.sum(np.triu(np.ones_like(prepare_matrix, dtype=bool)))
    print(f"Valid entries: {valid_count} out of {prepare_matrix.size}")

    # Create single figure
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Set view angle to match viz_3d.py exactly
    ax.view_init(elev=25, azim=200)
    
    # Adjust aspect ratio to match viz_3d.py
    ax.set_box_aspect([1, 1.5, 1.2])

    # Find global maximum for consistent scaling
    global_max_z = max(np.max(prepare_matrix), np.max(total_compute_matrix))

    # Prepare coordinates - only for valid entries (upper triangle, transposed)
    # x corresponds to start_vals (columns), y corresponds to sparse_ratios (rows)
    x, y = np.meshgrid(np.arange(len(start_vals)), np.arange(len(sparse_ratios)))
    
    bar_width = 0.5
    bar_depth = 0.5
    offset = 0.5

    # Draw bars column by column (for proper z-ordering with upper triangle)
    # For column j, valid entries are in rows 0 to j (upper triangle: row >= col)
    for j in reversed(range(H)):
        # Get valid rows for this column (rows where row_idx >= col_idx)
        rows_in_col = np.arange(j + 1)  # [0, 1, ..., j]

        # Skip if no valid entries (shouldn't happen with upper triangle, but safety check)
        if len(rows_in_col) == 0:
            continue

        # Coordinates for this column
        x_col = x[rows_in_col, j]
        y_col = y[rows_in_col, j]

        # Concatenate prepare and compute bars for single bar3d call
        # Compute bars: offset in positive y direction
        y_compute = y_col * scale + offset - 0.5
        # Prepare bars: offset in negative y direction
        y_prepare = y_col * scale - 0.5

        # Concatenate all coordinates and heights
        x_combined = np.concatenate([x_col, x_col])
        y_combined = np.concatenate([y_compute, y_prepare])
        z_combined = np.zeros_like(x_combined)
        dz_combined = np.concatenate([total_compute_matrix[rows_in_col, j], prepare_matrix[rows_in_col, j]])
        colors_combined = [compute_color] * len(x_col) + [prepare_color] * len(x_col)

        colors_transparent = [matplotlib.colors.colorConverter.to_rgb(compute_color) + (0.0, )] * len(x_col) + [matplotlib.colors.colorConverter.to_rgb(prepare_color) + (0.0, )] * len(x_col)
        
        # Single bar3d call for both bar types in this column
        ax.bar3d(x_combined, y_combined, z_combined,
                 bar_width, bar_depth, dz_combined,
                 color=colors_combined,
                 shade=False,
                 zsort='average')
        
        ax.bar3d(x_combined, y_combined, z_combined,
                 bar_width, bar_depth, dz_combined,
                 color=colors_transparent,
                 edgecolor='black',
                 linewidth=0.3,
                 shade=False,
                 zsort='average')
    
    # Set labels (swapped due to transpose)
    ax.set_xlabel('Max Sparsity', labelpad=15)
    ax.set_ylabel('Overal Sparsity', labelpad=18)
    ax.set_zlabel('Latency (ms)', labelpad=10)

    # Set ticks (swapped due to transpose)
    ax.set_xticks(np.arange(len(start_vals)))
    ax.set_xticklabels([f'{s:.1f}' for s in start_vals])
    ax.set_yticks(np.arange(len(sparse_ratios)) * scale)
    ax.set_yticklabels([f'{r:.1f}' for r in sparse_ratios])

    # Hide grid
    ax.grid(False)

    # Set axis limits for manual frame drawing (swapped due to transpose)
    x_min, x_max = -0.5, len(start_vals) - 0.5
    y_min, y_max = -0.5 * scale, (len(sparse_ratios) - 0.5) * scale
    z_min, z_max = 0, global_max_z * 1.2

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Add manual wireframe frame (12 edges of the cube)
    # Bottom frame edges (4 edges) - z=0 plane
    ax.plot([x_min, x_max], [y_min, y_min], [z_min, z_min], 'k-', linewidth=1.5, zorder=100)
    ax.plot([x_max, x_max], [y_min, y_max], [z_min, z_min], 'k-', linewidth=1.5, zorder=-100)
    ax.plot([x_max, x_min], [y_max, y_max], [z_min, z_min], 'k-', linewidth=1.5, zorder=-100)
    ax.plot([x_min, x_min], [y_max, y_min], [z_min, z_min], 'k-', linewidth=1.5, zorder=100)

    # Top frame edges (4 edges) - z=max plane
    ax.plot([x_min, x_max], [y_min, y_min], [z_max, z_max], 'k-', linewidth=1.2, alpha=0.8, zorder=100)
    ax.plot([x_max, x_max], [y_min, y_max], [z_max, z_max], 'k-', linewidth=1.2, alpha=0.8, zorder=-100)
    ax.plot([x_max, x_min], [y_max, y_max], [z_max, z_max], 'k-', linewidth=1.2, alpha=0.8, zorder=-100)
    ax.plot([x_min, x_min], [y_max, y_min], [z_max, z_max], 'k-', linewidth=1.2, alpha=0.8, zorder=100)

    # Vertical edges (4 edges) - connecting bottom to top
    ax.plot([x_min, x_min], [y_min, y_min], [z_min, z_max], 'k-', linewidth=2.0, alpha=0.9, zorder=100)
    ax.plot([x_max, x_max], [y_min, y_min], [z_min, z_max], 'k-', linewidth=1.8, alpha=0.9, zorder=-100)
    ax.plot([x_max, x_max], [y_max, y_max], [z_min, z_max], 'k-', linewidth=1.4, alpha=0.7, zorder=-100)
    ax.plot([x_min, x_min], [y_max, y_max], [z_min, z_max], 'k-', linewidth=1.2, alpha=0.6, zorder=-100)

    # Intermediate horizontal levels (for tick marks)
    num_z_ticks = 5
    for i in range(1, num_z_ticks):
        z_level = (z_max * i) / num_z_ticks
        alpha_val = 0.5 - (i * 0.08)
        linewidth_val = 0.8 if i % 2 == 1 else 0.5

        ax.plot([x_min, x_max], [y_min, y_min], [z_level, z_level],
                'k-', linewidth=linewidth_val, alpha=alpha_val, zorder=100)
        ax.plot([x_max, x_max], [y_min, y_max], [z_level, z_level],
                'k-', linewidth=linewidth_val, alpha=alpha_val, zorder=-100)
        ax.plot([x_max, x_min], [y_max, y_max], [z_level, z_level],
                'k-', linewidth=linewidth_val, alpha=alpha_val, zorder=-100)
        ax.plot([x_min, x_min], [y_max, y_min], [z_level, z_level],
                'k-', linewidth=linewidth_val, alpha=alpha_val, zorder=100)    
    
    num_y_ticks = 7
    for i in range(1, num_y_ticks):
        y_level = (i - 0.5) * scale
        linewidth_val = 0.8 if i % 2 == 1 else 0.5

        ax.plot([x_min, x_max], [y_level, y_level], [z_min, z_min],
                'k-', linewidth=linewidth_val, zorder=100)
        ax.plot([x_max, x_max], [y_level, y_level], [z_min, z_max],
                'k-', linewidth=linewidth_val, zorder=100)
        ax.plot([x_max, x_min], [y_level, y_level], [z_max, z_max],
                'k-', linewidth=linewidth_val, zorder=100)
        ax.plot([x_min, x_min], [y_level, y_level], [z_max, z_min],
                'k-', linewidth=linewidth_val, zorder=100)    

    # Clean up pane colors
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=prepare_color, label='Prep. Latency'),
        Patch(facecolor=compute_color, label='Comp. Latency')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # Final adjustments to use the whole canvas (same as viz_3d.py)
    # plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    plt.savefig(f"./figs/sparse_latency_breakdown.{file_format}", dpi=300) # , bbox_inches='tight')
    print(f"Merged breakdown visualization saved as 'sparse_latency_breakdown.{file_format}'")

    return fig

if __name__ == "__main__":
    print("Generating sparse latency breakdown visualization...")

    # Create the breakdown visualization
    fig = create_latency_breakdown()

    print("Breakdown visualization completed!")