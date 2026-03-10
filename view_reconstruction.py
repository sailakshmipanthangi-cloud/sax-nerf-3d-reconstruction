import numpy as np
import matplotlib.pyplot as plt

# Load reconstruction volume
data = np.load("logs/tensorf/chest_50/2026_03_10_08_23_48/eval/epoch_00000/image_pred.npy")

print("Volume shape:", data.shape)

# choose middle slice
slice_index = data.shape[0] // 2
slice_img = data[slice_index]

# normalize contrast
slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())

plt.figure(figsize=(6,6))
plt.imshow(slice_img, cmap="gray")
plt.title(f"Reconstructed CT Slice {slice_index}")
plt.axis("off")

plt.show()