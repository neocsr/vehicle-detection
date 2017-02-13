import cv2
import numpy as np
import matplotlib.image as mpimg
from skimage.feature import hog


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()

    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features


# Define a function to return HOG features and visualization
def get_hog_features(img, orientations, pix_per_cell, cell_per_block,
                     visualise=False, feature_vector=True):

    pixels_per_cell = (pix_per_cell, pix_per_cell)
    cells_per_block = (cell_per_block, cell_per_block)

    if visualise is True:
        features, hog_image = hog(img,
                                  orientations=orientations,
                                  pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block,
                                  transform_sqrt=True,
                                  visualise=visualise,
                                  feature_vector=feature_vector)
        return features, hog_image

    else:
        features = hog(img,
                       orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       transform_sqrt=True,
                       visualise=visualise,
                       feature_vector=feature_vector)
        return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB',
                     spatial_size=(32, 32), hist_bins=32, orientations=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):

    features = []

    for file in imgs:
        file_features = []

        image = mpimg.imread(file)

        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat is True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat is True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat is True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(
                        get_hog_features(feature_image[:, :, channel],
                                         orientations,
                                         pix_per_cell, cell_per_block,
                                         visualise=False, feature_vector=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(
                        feature_image[:, :, hog_channel],
                        orientations,
                        pix_per_cell, cell_per_block,
                        visualise=False, feature_vector=True)

            file_features.append(hog_features)

        features.append(np.concatenate(file_features))

    return features


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orientations=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    img_features = []

    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)

    if spatial_feat is True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    if hist_feat is True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    if hog_feat is True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(
                    get_hog_features(feature_image[:, :, channel],
                                     orientations,
                                     pix_per_cell, cell_per_block,
                                     visualise=False,
                                     feature_vector=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                            orientations,
                                            pix_per_cell, cell_per_block,
                                            visualise=False,
                                            feature_vector=True)

        img_features.append(hog_features)

    return np.concatenate(img_features)
