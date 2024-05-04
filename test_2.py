import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
fig.canvas.draw()

# Convert the canvas to a raw RGB buffer
buf = fig.canvas.tostring_rgb()
ncols, nrows = fig.canvas.get_width_height()
image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

print("Image shape:", image.shape)
print("First 3x3 pixels and RGB values:")
print(image[:3, :3, :])