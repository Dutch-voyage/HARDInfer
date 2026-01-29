import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import BboxBase, Bbox
import json

is_last_subplot=False

# Load latency stats
with open('HA_latency_summary.json', 'r') as f:
    latency_stats = json.load(f)

def cropped(self, pad):
    """
    Construct a `Bbox` by padding this one on all four sides.

    Parameters
    ----------
    w_pad : float
        Width pad
    h_pad : float, optional
        Height pad.  Defaults to *w_pad*.

    """
    points = self.get_points()
    return Bbox(points + [[-pad[0], -pad[2]], [pad[1], pad[3]]])

BboxBase.cropped = cropped

def generate_matrix(start_val, sparse_ratio):
    """
    Generate a 2D matrix with the given start_val and sparse_ratio.

    Returns:
        matrix: 2D numpy array
        r1, r2: the computed ratio values
    """
    rows, cols = 36, 8
    target_sum = 8 * 36 * sparse_ratio
    low, high = 0.0, 0.9999
    r1 = 0.5
    k = 2

    for _ in range(100):
        r1 = (low + high) / 2
        r2 = r1 ** k

        sum_r1 = (1 - r1**rows) / (1 - r1)
        sum_r2 = (1 - r2**cols) / (1 - r2)
        current_sum = start_val * sum_r1 * sum_r2

        if abs(current_sum - target_sum) < 1e-9:
            break
        elif current_sum < target_sum:
            low = r1
        else:
            high = r1

    r2 = r1 ** k

    # Generate the actual matrix
    matrix = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = start_val * (r1**i) * (r2**j)

    return matrix, r1, r2

def custom_color_map(value):
    start_color = "#94becf"
    end_color = "#f9bebe"
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_cmap", [start_color, end_color])
    return cmap(value)

def create_single_3d_plot(ax, matrix, start_val, sparse_ratio, title_prefix="", show_z_ticks=True, global_max_z=None, latency_info=None):
    """
    Create a single 3D plot on the given axes.

    Args:
        latency_info: Dict with 'sparse' and 'max' keys, each containing 'prep_ms' and 'compute_ms'
    """
    rows, cols = matrix.shape

    # Set aspect ratio for better 3D perception
    ax.set_box_aspect([1, 2.5, 1])  # [x, y, z] ratio

    # Set pure white background
    ax.xaxis.set_pane_color((1, 1, 1, 1))
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    # Create coordinate arrays
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    for i in range(rows):

        # Flatten the coordinate arrays for bar positions
        x_pos = X[i, :].flatten()
        y_pos = Y[i, :].flatten()
        z_pos = np.zeros_like(x_pos)

        # Get the heights (matrix values)
        heights = matrix[i, :].flatten()
        local_max_height = np.max(heights)

        # Color Scheme - use global max for consistent scaling if provided
        # academic_colormap = 'Blues'
        # cmap_obj = plt.colormaps[academic_colormap]
        cmap_obj = custom_color_map
        if global_max_z is not None:
            transparent_colors = [matplotlib.colors.colorConverter.to_rgb(color_item) + (0.0, ) for color_item in cmap_obj(heights / global_max_z)]
            colors = [matplotlib.colors.colorConverter.to_rgb(color_item) + (1.0, ) for color_item in cmap_obj(heights / global_max_z)]
        else:
            transparent_colors = [matplotlib.colors.colorConverter.to_rgb(color_item) + (0.0, ) for color_item in cmap_obj(heights / local_max_height)]
            colors = [matplotlib.colors.colorConverter.to_rgb(color_item) + (1.0, ) for color_item in cmap_obj(heights / local_max_height)]

        # Create the 3D bars
        dx = 0.80
        dy = 0.80

        # Render bars with NO edges first
        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, heights, color=colors, shade=False, zsort="max")
        
        # Render bars with NO edges first
        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, heights, color=transparent_colors, linewidth=0.5, 
                 edgecolor='black', shade=False, zsort="max")
        
        

    # Customize the plot - match actual axis display behavior
    # X-axis shows Head Id (0-7), Y-axis shows Layer Id (0-36), Z-axis shows values

    ax.set_xlabel('Head Id', labelpad=4, fontsize=10)  # Closer to axis
    ax.set_ylabel('Layer Id', labelpad=4, fontsize=10)

    # Only show z-label on the first subplot
    if show_z_ticks:
        ax.set_zlabel('Sparse Ratio', labelpad=4, fontsize=10)
    else:
        ax.set_zlabel('')

    # Text annotation for parameters with latency stats
    if latency_info is not None:
        sparse_stats = latency_info['sparse']
        max_stats = latency_info['max']
        annotation_text = (f'Max Sparsity per head: {start_val}\n'
                           f'Overall Sparsity: {sparse_ratio:.3f}\n'
                           f'Sparity Utilization: {sparse_ratio/start_val * 100:.0f}%\n'
                        #    f'HA-sparse:\n'
                        #    f'  Prep: {sparse_stats["prep_ms"]:.3f}ms\n'
                        #    f'  Compute (36L): {sparse_stats["compute_ms"]:.2f}ms\n'
                           f'HA-max:\n'
                           f'  Prep: {max_stats["prep_ms"]:.3f}ms\n'
                           f'  Compute (36L): {max_stats["compute_ms"]:.2f}ms')
    else:
        annotation_text = f'Max Sparsity per head: {start_val}, \nOverall Sparsity: {sparse_ratio:.3f}'

    ax.annotate(annotation_text,
                xy=(0.80, 0.70), xycoords='axes fraction',
                ha='right', va='top', fontsize=9, fontweight='bold', zorder=200,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.7))

    # Show x and y ticks, hide z ticks - use wireframe lines as visual ticks instead
    ax.set_xticks(range(0, cols, 1))
    ax.set_yticks(range(0, rows, 3))
    ax.set_xticklabels([str(i) for i in range(0, cols, 1)], fontsize=8)
    ax.set_yticklabels([str(i) for i in range(0, rows, 3)], fontsize=8)
    if not show_z_ticks:
        ax.set_zticks([])  # Remove z-ticks for non-first subplots

    # Hide grid
    ax.grid(False)

    # Set axis limits - use global max for wireframe ticks
    x_min, x_max = 0, cols
    y_min, y_max = 0, rows
    z_min = 0  # Always start from 0
    z_max = global_max_z if global_max_z is not None else local_max_height  # Use global for wireframe

    # Set axis limits with adequate padding within 3D space
    ax.set_xlim(-0.5, cols + 1)  # Good padding within 3D figure
    ax.set_ylim(-1, rows + 0.5)  # Good padding within 3D figure
    ax.set_zlim(0, z_max * 1.05)  # Small padding at top

    # Add wireframe lines as visual ticks - using global max for consistent tick positions
    # Complete 12-edge cube wireframe that works well with rotated views
    num_z_ticks = 8  # Number of intermediate z-level ticksax.bar3d(x_pos, y_pos, z_pos, dx, dy, heights, alpha=1, color=colors, linewidth=0.5, edgecolor='black', shade=False)

    # ===== MAIN CUBE FRAME (12 EDGES TOTAL) =====

    # Bottom frame edges (4 edges) - z=0 plane
    ax.plot([x_min, x_max], [y_min, y_min], [z_min, z_min], 'k-', linewidth=1.5, zorder=100, label='Bottom Front')
    # Right edge (x_max)
    ax.plot([x_max, x_max], [y_min, y_max], [z_min, z_min], 'k-', linewidth=1.5, label='Bottom Right')
    # Back edge (y_max)
    ax.plot([x_max, x_min], [y_max, y_max], [z_min, z_min], 'k-', linewidth=1.5, label='Bottom Back')
    # Left edge (x_min)
    ax.plot([x_min, x_min], [y_max, y_min], [z_min, z_min], 'k-', linewidth=1.5, label='Bottom Left')

    # Top frame edges (4 edges) - z=global max plane
    ax.plot([x_min, x_max], [y_min, y_min], [z_max, z_max], 'k-', linewidth=1.2, alpha=0.8, zorder=100, label='Top Front')
    # Right edge (x_max)
    ax.plot([x_max, x_max], [y_min, y_max], [z_max, z_max], 'k-', linewidth=1.2, alpha=0.8, zorder=100, label='Top Right')
    # Back edge (y_max)
    ax.plot([x_max, x_min], [y_max, y_max], [z_max, z_max], 'k-', linewidth=1.2, alpha=0.8, zorder=100, label='Top Back')
    # Left edge (x_min)
    ax.plot([x_min, x_min], [y_max, y_min], [z_max, z_max], 'k-', linewidth=1.2, alpha=0.8, zorder=100, label='Top Left')

    # Vertical edges (4 edges) - connecting bottom to top
    ax.plot([x_min, x_min], [y_min, y_min], [z_min, z_max], 'k-', linewidth=2.0, alpha=0.9, label='Vertical Front-Left')
    # Front-right vertical edge
    ax.plot([x_max, x_max], [y_min, y_min], [z_min, z_max], 'k-', linewidth=1.8, alpha=0.9, zorder=100, label='Vertical Front-Right')
    # Back-right vertical edge
    ax.plot([x_max, x_max], [y_max, y_max], [z_min, z_max], 'k-', linewidth=1.4, alpha=0.7, label='Vertical Back-Right')
    # Back-left vertical edge
    ax.plot([x_min, x_min], [y_max, y_max], [z_min, z_max], 'k-', linewidth=1.2, alpha=0.6, label='Vertical Back-Left')

    # ===== INTERMEDIATE HORIZONTAL LEVELS (for tick marks) =====
    for i in range(1, num_z_ticks):
        z_level = (z_max * i) / num_z_ticks
        alpha_val = 0.5 - (i * 0.04)  # Decreasing opacity for higher levels
        linewidth_val = 0.8 if i % 2 == 1 else 0.5  # Alternate line thickness

        # Horizontal rectangle at each intermediate z-level
        ax.plot([x_min, x_max], [y_min, y_min], [z_level, z_level],
                'k-', linewidth=linewidth_val, alpha=alpha_val, zorder=100)
        ax.plot([x_max, x_max], [y_min, y_max], [z_level, z_level],
                'k-', linewidth=linewidth_val, alpha=alpha_val, zorder=100)
        ax.plot([x_max, x_min], [y_max, y_max], [z_level, z_level],
                'k-', linewidth=linewidth_val, alpha=alpha_val)
        ax.plot([x_min, x_min], [y_max, y_min], [z_level, z_level],
                'k-', linewidth=linewidth_val, alpha=alpha_val)

    return local_max_height

def visualize_3d_matrix_multi(settings, title_prefix=""):
    """
    Generate multiple 2D matrices and visualize them as 3D columns in subfigures.

    Args:
        settings: List of tuples (start_val, sparse_ratio) for each subplot
        title_prefix: Optional prefix for subplot titles
    """
    num_subplots = len(settings)

    # Map settings to their corresponding latency stats
    setting_to_stats = {
        (0.5, 1/8): "HA_backend_stats_0.5_0.125.npy",
        (0.5, 1/4): "HA_backend_stats_0.5_0.25.npy",
        (0.25, 1/8): "HA_backend_stats_0.25_0.125.npy",
    }

    # Create figure with equal subplot widths using GridSpec
    # All subplots have same width for consistent appearance
    width_ratios = [1.0] * num_subplots

    # Create figure with global margins
    fig = plt.figure(figsize=(14, 7), facecolor='white')
    # Set global subplot margins for the entire figure
    fig.subplots_adjust(left=0.05, right=1, top=0.95, bottom=0.08, wspace=-0.05)

    # Create GridSpec without individual margins (they're set globally)
    gs = GridSpec(1, num_subplots, figure=fig, width_ratios=width_ratios)

    # Generate all matrices first to determine global z-max for consistent scaling
    matrices = []
    max_z_vals = []
    for start_val, sparse_ratio in settings:
        matrix, _, _ = generate_matrix(start_val, sparse_ratio)
        matrices.append(matrix)
        max_z_vals.append(np.max(matrix))

    global_max_z = max(max_z_vals)

    # Create subplots using GridSpec for variable widths
    for i, ((start_val, sparse_ratio), matrix) in enumerate(zip(settings, matrices)):
        ax = fig.add_subplot(gs[0, i], projection='3d')

        ax.view_init(elev=25, azim=-35)

        # Only show z-ticks on the first subplot
        show_z_ticks = (i == 2)

        # Get latency info for this setting
        latency_info = None
        setting_key = (start_val, sparse_ratio)
        if setting_key in setting_to_stats:
            stat_file = setting_to_stats[setting_key]
            if stat_file in latency_stats:
                sparse_prefill_stats = latency_stats[stat_file]['HA_sparse_prefill']
                max_prefill_stats = latency_stats[stat_file]['HA_flatten_max_prefill']
                latency_info = {
                    'sparse': {
                        'prep_ms': sparse_prefill_stats['preparation_time_avg_ms'],
                        'compute_ms': sparse_prefill_stats['compute_time_36_layers_ms']
                    },
                    'max': {
                        'prep_ms': max_prefill_stats['preparation_time_avg_ms'],
                        'compute_ms': max_prefill_stats['compute_time_36_layers_ms']
                    }
                }

        # Create the 3D plot with global max z for consistent wireframe ticks
        create_single_3d_plot(ax, matrix, start_val, sparse_ratio, title_prefix, show_z_ticks, global_max_z, latency_info)

        # Set consistent z-axis limits across all subplots
        ax.set_zlim(0, global_max_z * 1.05)  # Add 5% margin

        # Set z-ticks only on the first subplot
        if show_z_ticks:
            ax.set_zticks(np.linspace(0, global_max_z, 5))

        # Set consistent viewing angle
        
        print(f"Subplot {i+1}: start_val={start_val}, sparse_ratio={sparse_ratio}, max_z={np.max(matrix):.4f}")

    # GridSpec already handles spacing, no need for additional adjustment
    
    # Save the figure with asymmetric padding
    output_file = './figs/sparsity_utilizations.png'
    # Get current figure bounds and add asymmetric padding
    renderer = fig.canvas.get_renderer()
    fig.draw(renderer)
    bbox = fig.get_tightbbox(renderer)

    # Add different padding for x and y directions (in inches)
    # pad_x: horizontal padding, pad_y: vertical padding
    pad_x = 0.28  # Less horizontal padding
    pad_y = - 0.35 # More vertical padding

    # Expand bbox with asymmetric padding
    bbox_expanded = bbox.padded(pad_x, pad_y)
    bbox_expanded = bbox.cropped((0.28, 0.28, -0.45, -0.60))

    plt.savefig(output_file, dpi=300, bbox_inches=bbox_expanded)
    print(f"\nMulti-panel 3D visualization saved to: {output_file}")

    # Also show statistics
    print(f"\nMatrix Statistics:")
    for i, ((start_val, sparse_ratio), matrix) in enumerate(zip(settings, matrices)):
        print(f"  Subplot {i+1} (start={start_val}, sparse={sparse_ratio:.3f}):")
        print(f"    Matrix shape: {matrix.shape}")
        print(f"    Min value: {np.min(matrix):.6f}")
        print(f"    Max value: {np.max(matrix):.6f}")
        print(f"    Mean value: {np.mean(matrix):.6f}")
        print(f"    Total sum: {np.sum(matrix):.6f}")

    # Display the plot
    plt.show()

if __name__ == "__main__":
    print("\n=== Multi-Plot Example ===")
    settings = [
        (0.5, 1/8),
        (0.5, 1/4),
        (0.25, 1/8)
    ]

    # Run multi-panel visualization
    visualize_3d_matrix_multi(settings, title_prefix="Config ")
