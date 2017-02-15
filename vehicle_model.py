import glob
import time
import numpy as np
import pickle
from hashlib import md5
from os import path
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from vehicle_features import extract_features


def build_model(color_space='RGB', orientations=9,
                pix_per_cell=8, cell_per_block=2, hog_channel=0,
                spatial_size=(16, 16), hist_bins=16,
                spatial_feat=True, hist_feat=True, hog_feat=True):

    string = '{}{}{}{}{}{}{}{}{}{}'.format(color_space, orientations,
                                           pix_per_cell, cell_per_block,
                                           hog_channel, spatial_size,
                                           hist_bins, spatial_feat,
                                           hist_feat, hog_feat)
    hsh = md5(str.encode(string)).hexdigest()
    svc_filename = 'svc_{}.p'.format(hsh)
    scaler_filename = 'scaler_{}.p'.format(hsh)

    if path.isfile(svc_filename) and path.isfile(scaler_filename):
        print('Using existing files "{}" and "{}"...'.format(svc_filename,
                                                             scaler_filename))
        with open(scaler_filename, 'rb') as file:
            X_scaler = pickle.load(file)

        with open(svc_filename, 'rb') as file:
            svc = pickle.load(file)

        return (svc, X_scaler)

    # Large Dataset
    # =============
    # Load cars and not cars datasets
    cars = glob.glob('data/vehicles_largeset/**/*.png')
    notcars = glob.glob('data/non-vehicles_largeset/**/*.png')
    print('Loading {} car images...'.format(len(cars)))
    print('Loading {} not car images...'.format(len(notcars)))

    # Small Dataset
    # =============
    # # Load cars and not cars datasets
    # images = glob.glob('data/*_smallset/**/*.jpeg')
    # print('Loading {} images...'.format(len(images)))

    # cars = []
    # notcars = []

    # for image in images:
    #     if 'image' in image or 'extra' in image:
    #         notcars.append(image)
    #     else:
    #         cars.append(image)

    # # Reduce the sample size because HOG features are slow to compute
    # # The quiz evaluator times out after 13s of CPU time
    # sample_size = 2000
    # cars = cars[0:sample_size]
    # notcars = notcars[0:sample_size]

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size,
                                    hist_bins=hist_bins,
                                    orientations=orientations,
                                    pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel,
                                    spatial_feat=spatial_feat,
                                    hist_feat=hist_feat,
                                    hog_feat=hog_feat)

    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins,
                                       orientations=orientations,
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
                                       spatial_feat=spatial_feat,
                                       hist_feat=hist_feat,
                                       hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = 1  # np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orientations, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()

    # Training
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Save the model
    print('Saving "{}" and "{}" files'.format(svc_filename, scaler_filename))

    with open(scaler_filename, 'wb') as file:
        pickle.dump(X_scaler, file)

    with open(svc_filename, 'wb') as file:
        pickle.dump(svc, file)

    return (svc, X_scaler)
