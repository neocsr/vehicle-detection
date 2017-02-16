import numpy as np
import time
from scipy.ndimage.measurements import label
from vehicle_search import add_heat, apply_threshold
from vehicle_search import draw_boxes, draw_labeled_bboxes
from vehicle_search import search_windows
from vehicle_tracker import Tracker

tracker = Tracker()

def process_image(img, windows, clf, scaler,
                  color_space='RGB', spatial_size=(16, 16),
                  hist_bins=16, orientations=9,
                  pix_per_cell=8, cell_per_block=2,
                  hog_channel=0, spatial_feat=True,
                  hist_feat=True, hog_feat=True):
    draw_image = np.copy(img)

    # Check the prediction time for searching each image
    # t = time.time()

    hot_windows = search_windows(img,
                                 windows=windows,
                                 clf=clf,
                                 scaler=scaler,
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
    # t2 = time.time()

    # print(round(t2-t, 2), 'seconds to search windows...')

    tracker.add_hot_windows(hot_windows)

    last_hot_windows = tracker.get_last_hot_windows()[0]

    window_img = draw_boxes(draw_image, last_hot_windows,
                            color=(255, 80, 0), thickness=6)

    heatmap_img = np.zeros_like(img[:, :, 0]).astype(np.float)
    add_heat(heatmap=heatmap_img, bbox_list=last_hot_windows)

    labels = label(heatmap_img)
    heatmap_img = apply_threshold(heatmap_img, 2)
    labels = label(heatmap_img)

    # print(labels[1], 'cars found')

    labeled_img = draw_labeled_bboxes(np.copy(img),
                                      labels=labels, color=(255, 80, 0))

    return (window_img, heatmap_img, labeled_img)
