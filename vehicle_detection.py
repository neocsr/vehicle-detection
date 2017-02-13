import numpy as np
import cv2
import glob
import time
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from moviepy.editor import VideoFileClip
from vehicle_features import extract_features
from vehicle_search import add_heat, apply_threshold
from vehicle_search import draw_boxes, draw_labeled_bboxes
from vehicle_search import search_windows, slide_window
from vehicle_model import build_model

# Load sample images
sample_files = glob.glob('test_images/*.jpg')
sample_images = []

for file in sample_files:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sample_images.append(img)

img = np.copy(sample_images[0])
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# TODO: Tweak these parameters and see how the results change.
color_space = 'LUV'           # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orientations = 9              # HOG orientations
pix_per_cell = 8              # HOG pixels per cell
cell_per_block = 2            # HOG cells per block
hog_channel = 'ALL'           # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)       # Spatial binning dimensions
hist_bins = 16                # Number of histogram bins
spatial_feat = True           # Spatial features on or off
hist_feat = True              # Histogram features on or off
hog_feat = True               # HOG features on or off

svc, X_scaler = build_model(color_space=color_space,
                            orientations=orientations,
                            pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel,
                            spatial_size=spatial_size,
                            hist_bins=hist_bins,
                            spatial_feat=spatial_feat,
                            hist_feat=hist_feat,
                            hog_feat=hog_feat)

image = np.copy(img)
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255

y_start_stop = [int(img.shape[0]*1/3), None]   # Min and max in y to search

windows = slide_window(image,
                       x_start_stop=[None, None],
                       y_start_stop=y_start_stop,
                       xy_window=(96, 96),
                       xy_overlap=(0.5, 0.5))

# Check the prediction time for searching each image
t = time.time()

hot_windows = search_windows(image, windows=windows, clf=svc, scaler=X_scaler,
                             color_space=color_space,
                             spatial_size=spatial_size,
                             hist_bins=hist_bins,
                             orientations=orientations,
                             pix_per_cell=pix_per_cell,
                             cell_per_block=cell_per_block,
                             hog_channel=hog_channel,
                             spatial_feat=spatial_feat,
                             hist_feat=hist_feat,
                             hog_feat=hog_feat)
t2 = time.time()

print(round(t2-t, 2), 'Seconds to search windows...')


window_img = draw_boxes(draw_image, hot_windows,
                        color=(0, 0, 255), thickness=6)

plt.figure(figsize=(10, 6))
plt.imshow(window_img)
plt.show()

heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
add_heat(heatmap=heatmap, bbox_list=hot_windows)
labels = label(heatmap)
heatmap = apply_threshold(heatmap, 0)
labels = label(heatmap)

print(labels[1], 'cars found')

plt.figure(figsize=(10, 6))
plt.imshow(labels[0], cmap='hot')
plt.show()

draw_img = draw_labeled_bboxes(np.copy(img), labels)

plt.figure(figsize=(10, 6))
plt.imshow(draw_img)
plt.axis('off')
plt.show()
