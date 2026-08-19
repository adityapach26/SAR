import numpy as np
from scipy.ndimage import uniform_filter


def build_multichannel_input(vv_array_uint8, kernel_size=7):
    """
    Build a 3-channel input from a single-channel VV uint8 array.

    Parameters
    ----------
    vv_array_uint8 : np.ndarray
        2D array of dtype uint8, shape (H, W).
    kernel_size : int, optional
        Size of the uniform filter for texture computation (default 7).

    Returns
    -------
    np.ndarray
        Float32 array of shape (3, H, W) with channels:
        0: VV normalized to [0,1]
        1: log1p(VV) normalized to [0,1]
        2: local texture (windowed std) normalized to [0,1]
    """
    # Ensure input is 2D
    if vv_array_uint8.ndim != 2:
        raise ValueError("Input array must be 2D (H, W)")

    # Convert to float32 in [0, 1]
    vv_float = vv_array_uint8.astype(np.float32) / 255.0  # shape (H, W)

    # Channel 1: VV normalized
    chan_vv = vv_float

    # Channel 2: log-intensity normalized
    log_vv = np.log1p(vv_float)  # log(1 + x)
    # Min-max normalize to [0, 1]
    log_min = log_vv.min()
    log_max = log_vv.max()
    if log_max - log_min > 1e-8:
        chan_log = (log_vv - log_min) / (log_max - log_min)
    else:
        chan_log = np.zeros_like(log_vv)

    # Channel 3: local texture (windowed standard deviation)
    # Compute local mean and mean of squares using uniform_filter
    local_mean = uniform_filter(vv_float, size=kernel_size)
    local_sq_mean = uniform_filter(vv_float ** 2, size=kernel_size)
    local_var = local_sq_mean - local_mean ** 2
    # Avoid negative variance due to numerical errors
    local_var = np.maximum(local_var, 0.0)
    local_std = np.sqrt(local_var)
    # Min-max normalize texture to [0, 1]
    tex_min = local_std.min()
    tex_max = local_std.max()
    if tex_max - tex_min > 1e-8:
        chan_tex = (local_std - tex_min) / (tex_max - tex_min)
    else:
        chan_tex = np.zeros_like(local_std)

    # Stack channels: (3, H, W)
    multichannel = np.stack([chan_vv, chan_log, chan_tex], axis=0).astype(np.float32)
    return multichannel


if __name__ == "__main__":
    # Build a synthetic test image: smooth left half, noisy right half
    height, width = 256, 256
    smooth_val = 100  # uint8
    # Left half smooth
    left = np.full((height, width // 2), smooth_val, dtype=np.uint8)
    # Right half noisy uniform 0-255
    right = np.random.randint(0, 256, size=(height, width // 2), dtype=np.uint8)
    test_img = np.concatenate([left, right], axis=1)  # shape (H, W)

    # Run the function
    result = build_multichannel_input(test_img, kernel_size=7)
    # result shape (3, H, W)

    # Check output shape
    expected_shape = (3, height, width)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    print(f"Output shape check PASSED: {result.shape}")

    # Extract texture channel (index 2)
    texture_chan = result[2]  # (H, W)

    # Define masks for smooth and noisy regions
    smooth_mask = np.zeros_like(test_img, dtype=bool)
    smooth_mask[:, :width // 2] = True
    noisy_mask = ~smooth_mask

    # Compute mean texture in each region
    mean_smooth = texture_chan[smooth_mask].mean()
    mean_noisy = texture_chan[noisy_mask].mean()

    print(f"Smooth region mean texture: {mean_smooth:.6f}")
    print(f"Noisy region mean texture: {mean_noisy:.6f}")

    # Assert noisy region's mean texture is higher than smooth region's
    if mean_noisy > mean_smooth:
        print("TEST PASS: Noisy region has higher mean texture.")
    else:
        print("TEST FAIL: Noisy region does NOT have higher mean texture.")
        # Optionally raise AssertionError if you want to halt on fail
        # raise AssertionError("Texture test failed")

    # Plot the three channels side by side
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    channel_names = ["VV normalized", "log1p(VV) normalized", "Local texture (std)"]
    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        im = ax.imshow(result[i], cmap='viridis')
        ax.set_title(name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Multichannel input visualization")
    plt.tight_layout()
    output_fig_path = "texture_channels.png"
    plt.savefig(output_fig_path, dpi=150)
    print(f"Figure saved to {output_fig_path}")
    # Optionally show the plot (uncomment if running interactively)
    # plt.show()


    