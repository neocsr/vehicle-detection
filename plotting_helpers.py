import math
import matplotlib.pyplot as plt


def plot_images(images, cols=5, figsize=(8, 4), cmap='gray'):
    n_img = len(images)
    rows = math.ceil(n_img/cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    subplots = axes.ravel()

    for n, subplot in enumerate(subplots):
        if (n < n_img):
            image = images[n]
            if len(image.shape) < 3:
                subplots[n].imshow(image, cmap=cmap)
            else:
                subplots[n].imshow(image)
            subplots[n].axis('off')
        else:
            subplots[n].axis('off')
    plt.show()
