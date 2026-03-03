import matplotlib.pyplot as plt

def plot_maps(map_stack, rasterizer):
    layer_names = ["Land Dist", "Restricted Dist", "Ferry Dist", "Water Depth"]
    num_layers = map_stack.shape[0]
    
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))

    for i in range(num_layers):
        # Plotte das Raster (origin='lower' passt zum np.flip)
        im = axes[i].imshow(map_stack[i], origin='lower', cmap='viridis', extent=[0, rasterizer.x_size, 0, rasterizer.y_size])
        axes[i].set_title(layer_names[i])
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("map_validation_with_polygons.png")