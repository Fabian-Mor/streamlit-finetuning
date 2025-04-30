import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def plot_contour(u_vals, v_vals, z_vals, title, corner_labels=None, mask_triangle=False):
    u_mesh, v_mesh = np.meshgrid(u_vals, v_vals)

    fig, ax = plt.subplots(figsize=(6, 6 if mask_triangle else 5))  # Create figure and axis
    ax.set_aspect('equal' if mask_triangle else 'auto')

    if mask_triangle:
        # Triangular plot
        u_flat = u_mesh.flatten()
        v_flat = v_mesh.flatten()
        z_flat = z_vals.flatten()

        mask = (u_flat + v_flat) <= 1.0
        u_flat = u_flat[mask]
        v_flat = v_flat[mask]
        z_flat = z_flat[mask]

        x = v_flat + 0.5 * u_flat
        y = (np.sqrt(3) / 2) * u_flat

        triang = tri.Triangulation(x, y)

        contour = ax.tricontourf(triang, z_flat, levels=50, cmap='viridis')
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Value')

        ax.axis('off')

        if corner_labels is not None:
            for (u, v), label in corner_labels.items():
                x_pos = v + 0.5 * u
                y_pos = (np.sqrt(3) / 2) * u
                ax.text(x_pos, y_pos, label, fontsize=10, fontweight='bold', color='black', ha='center', va='center')

        ax.set_title(title, fontsize=14, pad=20)

    else:
        # Regular square plot
        if mask_triangle:
            mask = (u_mesh + v_mesh) > 1
            z_vals = np.ma.array(z_vals, mask=mask)

        contour = ax.contourf(u_mesh, v_mesh, z_vals, levels=50, cmap='viridis')
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Value')

        ax.set_xlabel('Parameter u')
        ax.set_ylabel('Parameter v')
        ax.set_title(title)

        min_idx = np.unravel_index(np.argmin(z_vals), z_vals.shape)
        max_idx = np.unravel_index(np.argmax(z_vals), z_vals.shape)
        ax.plot(u_mesh[min_idx], v_mesh[min_idx], 'ro', label='Min')
        ax.plot(u_mesh[max_idx], v_mesh[max_idx], 'go', label='Max')
        ax.text(u_mesh[min_idx], v_mesh[min_idx], 'Min', color='white', fontsize=8, ha='center', va='center')
        ax.text(u_mesh[max_idx], v_mesh[max_idx], 'Max', color='white', fontsize=8, ha='center', va='center')

        if corner_labels is not None:
            for (i, j), label in corner_labels.items():
                ax.text(u_mesh[i, j], v_mesh[i, j], label, fontsize=8, ha='center', va='bottom', color='white',
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))

        ax.legend()

    fig.tight_layout()
    return fig


def plot_PCA(
    explained_variance,
    base_point,
    finetuned_points,
    interpolated_points,
    losses_all,
    acc_all,
    acc=False,  # <- simple bool
):
    losses_finetuned = losses_all[1:1+len(finetuned_points)]
    losses_interpolated = losses_all[1+len(finetuned_points):]
    acc_finetuned = acc_all[1:1+len(finetuned_points)]
    acc_interpolated = acc_all[1+len(finetuned_points):]

    # Select values based on acc flag
    if acc:
        metric_values = acc_finetuned
        values_interp = acc_interpolated
        metric_name = "Accuracy"
        cmap_name = "viridis"
    else:
        metric_values = losses_finetuned
        values_interp = losses_interpolated
        metric_name = "Loss"
        cmap_name = "viridis"

    values_all = metric_values + values_interp
    norm = plt.Normalize(vmin=min(values_all), vmax=max(values_all))
    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(*zip(*([base_point] + list(finetuned_points))), '-o', label='Optimization Path', color='blue')
    ax.plot(*zip(*([base_point] + list(interpolated_points))), '--x', label='Interpolation Path', color='orange')

    # Color points
    for i, point in enumerate(finetuned_points):
        val = metric_values[i]
        ax.scatter(*point, color=cmap(norm(val)), s=100, zorder=5)
    for i, point in enumerate(interpolated_points):
        val = values_interp[i]
        ax.scatter(*point, color=cmap(norm(val)), s=100, zorder=5)

    # Start and end points
    ax.scatter(*base_point, color='green', s=100, label='Zeroshot Start', zorder=5)
    ax.scatter(*finetuned_points[-1], color='red', s=100, label='Finetuned End', zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(values_all)
    plt.colorbar(sm, ax=ax, label=metric_name)

    ax.set_title(
        f"2D PCA with {metric_name} (Explained Var: {explained_variance[0]:.2f}, {explained_variance[1]:.2f})")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return fig


def plot_training(losses, accs, acc=True):
    """
    losses: list of lists, each of length 11
    accs: list of lists, each of length 11
    acc: bool, if True plot accuracies, else losses
    """
    # Select data
    data = accs if acc else losses
    metric_name = "Accuracy" if acc else "Loss"

    num_epochs = len(data)
    if num_epochs == 0:
        raise ValueError("Empty data provided.")

    fig, ax = plt.subplots(figsize=(10, 6))

    epoch_numbers = np.arange(num_epochs)
    epoch_numbers_all = np.arange(num_epochs+1)

    # Prepare lists for clean plotting
    epoch_main_vals = [epoch_data[0] for epoch_data in data] + [data[-1][-1]]
    interp_vals = [epoch_data[1:-1] for epoch_data in data]

    # Plot main epoch points
    ax.scatter(epoch_numbers_all, epoch_main_vals,
            marker='o', color='blue',
            label=f'{metric_name} at Epoch'
            )

    # Plot interpolations
    for i in range(num_epochs):
        interp_x = np.linspace(epoch_numbers[i], epoch_numbers[i] + 1, len(interp_vals[i]) + 2)[1:-1]
        ax.scatter(interp_x, interp_vals[i], marker='x', color='gray', alpha=0.8,
                   label='Interpolation' if i == 0 else "")

    best_epoch_val = max(epoch_main_vals) if acc else min(epoch_main_vals)

    # Plot horizontal dashed red line at the best value
    ax.axhline(y=best_epoch_val, color='red', linestyle='--', alpha=0.5,
               label=f'Best {metric_name} ({best_epoch_val:.2f})')

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Training {metric_name} with Interpolations")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    return fig

def plot_compare(losses_finetuned, acc_finetuned,
                 losses_interpolated, acc_interpolated, acc=True):
    """
    losses_finetuned: list of losses for finetuned model
    acc_finetuned: list of accuracies for finetuned model
    losses_interpolated: list of losses for interpolated model
    acc_interpolated: list of accuracies for interpolated model
    acc: bool, if True plot accuracies, else losses
    """
    # Select data
    finetuned_data = acc_finetuned if acc else losses_finetuned
    interpolated_data = acc_interpolated if acc else losses_interpolated
    metric_name = "Accuracy" if acc else "Loss"

    num_points = len(finetuned_data)
    if num_points == 0 or len(interpolated_data) != num_points:
        raise ValueError("Data arrays must have the same length and not be empty.")

    fig, ax = plt.subplots(figsize=(10, 6))

    x_values = np.arange(num_points)

    # Plot finetuned values
    ax.plot(x_values, finetuned_data, marker='o', color='blue', label=f'Finetuned {metric_name}', alpha=0.8)

    # Plot interpolated values
    ax.plot(x_values, interpolated_data, marker='x', color='green', label=f'Interpolated {metric_name}', alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Comparison of {metric_name} (Finetuned vs Interpolated)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    return fig


def plot_triangle(coords, values, acc=True, corner_names=None):
    coords = np.array(coords)
    values = np.array(values)
    title = "Accuracy" if acc else "Loss"

    x = coords[:, 0]  # α (zeroshot)
    y = coords[:, 1]  # β (finetuned)
    z = values[:, 1] if acc else values[:, 0]

    triang = tri.Triangulation(x, y)

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.tricontourf(triang, z, levels=40, cmap="viridis")
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(title)

    # Add optimal point (max accuracy or min loss)
    best_idx = np.argmax(z) if acc else np.argmin(z)
    best_x, best_y = x[best_idx], y[best_idx]
    ax.plot(best_x, best_y, 'o', markersize=6, markeredgecolor='white',
            markerfacecolor='red', label="Best")

    # Add corner labels if provided
    if corner_names:
        ax.text(1.0, 0.0, corner_names[0], ha='right', va='top',
                fontsize=10, fontweight='bold', color='black')
        ax.text(0.0, 1.0, corner_names[1], ha='left', va='bottom',
                fontsize=10, fontweight='bold', color='black')
        ax.text(0.0, 0.0, corner_names[2], ha='left', va='top',
                fontsize=10, fontweight='bold', color='black')

    # Clean plot: remove ticks and axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', 'box')
    ax.axis('off')

    fig.tight_layout()
    return fig
