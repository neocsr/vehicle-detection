import numpy as np
import cv2
import glob
from moviepy.editor import VideoFileClip
from plotting_helpers import plot_images
from vehicle_search import generate_search_windows
from vehicle_model import build_model
from vehicle_processing import process_image
import matplotlib.pyplot as plt

# Load sample images
sample_files = glob.glob('test_images/*.jpg')
sample_images = []

for file in sample_files:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sample_images.append(img)

img = np.copy(sample_images[0])
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# plt.imshow(img)
# plt.grid('on')
# plt.savefig('img.jpg')
# plt.show()

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

windows = generate_search_windows(img)
output_imgs = []

for sample in sample_images:
    win_img, heat_img, label_img = process_image(sample,
                                                 windows=windows,
                                                 clf=svc,
                                                 scaler=X_scaler,
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
    output_imgs.append(win_img)
    output_imgs.append(heat_img)
    output_imgs.append(label_img)


# plot_images(output_imgs, cols=3, figsize=(10, 24), cmap='hot')


def process_frame(image):
    try:
        _, _, label_img = process_image(image,
                                        windows=windows,
                                        clf=svc,
                                        scaler=X_scaler,
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
        frame = label_img
    except Exception as e:
        frame = image
        print(e)
    finally:
        return frame

video_output = 'project_video_output.mp4'
clip = VideoFileClip('project_video.mp4')
# video_output = 'test_video_output.mp4'
# clip = VideoFileClip('test_video.mp4')
out_clip = clip.fl_image(process_frame)
out_clip.write_videofile(video_output, audio=False)
