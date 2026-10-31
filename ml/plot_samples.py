import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs('docs/samples', exist_ok=True)

CLASSES = ['boomerang', 'cat', 'eyeglasses', 'face', 'flower', 
           'hat','house', 'sailboat','spider', 'sword']

SAMPLES_PER_CLASS = 5

# create a figure with one row per class, one column per sample

fig, axes = plt.subplots(len(CLASSES), SAMPLES_PER_CLASS, figsize=(7, 7))

# loop over classes

for r, c in enumerate(CLASSES):

    data = np.load(f"data/{c}.npy", mmap_mode='r')
    idx = np.random.choice(len(data), SAMPLES_PER_CLASS, replace=False)

    for col, i in enumerate(idx):
        ax = axes[r, col]
        ax.imshow(data[i].reshape(28,28), cmap='gray_r')
        ax.axis('off')

    axes[r, 0].set_title(c, fontsize=8, loc='left')
    
# tight_layout, savefig, show

plt.tight_layout()              # fix spacing
plt.savefig('docs/samples/samples.png') # save it (good for your README)
plt.show()                      # open a window