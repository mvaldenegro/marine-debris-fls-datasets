import argparse
import numpy as np
import cv2

import matplotlib.pyplot as plt
import glob
import os
import h5py

from rectangle import Rectangle

def neighbors(p, image_size):
    ret = []

    if p[0] != 0:
        ret.append((p[0] - 1, p[1]))

    if p[1] != 0:
        ret.append((p[0], p[1] - 1))

    if p[0] + 1 < image_size[0]:
        ret.append((p[0] + 1, p[1]))

    if p[1] + 1 < image_size[1]:
        ret.append((p[0], p[1] + 1))

    return ret

from collections import deque

"""
Extracts a polar mask from a grayscale image by taking the assumption
that black pixels correspond to mask positions, but only if they are connected,
starting at (0, 0).
"""
def extract_polar_mask(image):
    visited = np.zeros(image.shape, dtype = np.uint8)

    queue = deque()
    queue.append((0, 0))

    while len(queue) > 0:
        y, x = queue.popleft()

        #Skip non-white pixels
        if image[y][x] < 200:
            continue

        if visited[y][x] > 0:
            continue

        visited[y][x] = 255

        for neighbor in neighbors((y, x), image.shape):
            if visited[neighbor[0]][neighbor[1]]:
                continue

            queue.append(neighbor)

    return (255 - visited).astype(np.bool)

"""
Runs a sliding window over a polar image, skipping all windows
that fall outside of the polar field of view.
Assumes that the image points up.
Computes all the sliding window rectangles and returns them in a list (list of Rectangle).
The polarMask parameter is a image that defines the polar FOV. This image
contains a 0 in pixels outside the FOV, and a value > 0 (usually 1) inside the FOV.
"""
def polar_sliding_window(imageSize, windowSize, polarMask, step_size = 2):
    actualPolarMask = polarMask

    windows = []

    xRange = imageSize[0] - windowSize[0]
    yRange = imageSize[1] - windowSize[1]

    for x in range(0, xRange, step_size):
        for y in range(0, yRange, step_size):

            #Check if it makes sense to start a window
            #at the current position
            if actualPolarMask[x][y] == 0:
                continue

            windowRect = Rectangle((x, y), windowSize[0], windowSize[1])

            #Check each corner fo the rectangle
            cornerX, cornerY = windowRect.topLeft
            if actualPolarMask[cornerX][cornerY] == 0:
                continue

            cornerX, cornerY = windowRect.topRight
            if actualPolarMask[cornerX][cornerY] == 0:
                continue

            cornerX, cornerY = windowRect.bottomLeft
            if actualPolarMask[cornerX][cornerY] == 0:
                continue

            cornerX, cornerY = windowRect.bottomRight
            if actualPolarMask[cornerX][cornerY] == 0:
                continue

            #And the center, just in case
            cornerX, cornerY = windowRect.center
            if actualPolarMask[cornerX][cornerY] == 0:
                continue

            #Ok, window is inside Sonar's FOV
            windows.append(windowRect)

    return windows

def filter_windows(windows, image, window_size=(96, 96), black_threshold = 20):
    ret_windows = []

    for window in windows:
        #Check for irregular sized windows
        if window.width != window_size[0]:
            continue

        if window.height != window_size[1]:
            continue

        #Check each corner fo the rectangle
        corner_x, corner_y = window.topLeft
        if image[corner_x][corner_y] < black_threshold:
            continue

        corner_x, corner_y = window.topRight
        if image[corner_x][corner_y] < black_threshold:
            continue

        corner_x, corner_y = window.bottomLeft
        if image[corner_x][corner_y] < black_threshold:
            continue

        corner_x, corner_y = window.bottomRight
        if image[corner_x][corner_y] < black_threshold:
            continue

        #And the center, just in case
        corner_x, corner_y = window.center
        if image[corner_x][corner_y] < black_threshold:
            continue

        #Ok, window is inside Sonar's FOV
        ret_windows.append(window)

    return ret_windows

def extract_windows(image, windows):
    extracted_patches = []

    for window in windows:
        x, y = window.topLeft
        w, h = window.width, window.height

        patch = image[x:x+w, y:y+h]
        extracted_patches.append(patch)

    return extracted_patches

def paint_windows(image, windows):
    painted_image = image.copy()
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_GRAY2BGR)

    for window in windows:
        x, y = window.topLeft
        w, h = window.width, window.height

        cv2.rectangle(painted_image, (y, x), (y + h, x + w), (255, 0, 0), 1)

    return painted_image

IMAGE_FOLDERS = [
    "2016-06-22_113541",
    "2016-06-22_114714",
    "2016-06-22_143000",
    "2016-06-22_143319",
    "2016-06-22_143818",
    "2016-06-22_144047",
    "2016-06-22_144358",
    "2016-06-22_144757",
    "2016-06-22_145239",
    "2016-06-22_145337"
]

HDF5_OUTPUT_PATTERN = "quarry-patches-{}x{}-stride-{}.hdf5"

if __name__ == "__main__":
    description = "Script to extract patches from full size FLS sonar images for the quarry scenario"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--patch_width", type=int, default=96, help="Width/height of extracted patches")
    parser.add_argument("--patch_height", type=int, default=96, help="Width/height of extracted patches")

    parser.add_argument("--patch_stride", type=int, default=4, help="Stride of candidate patch locations")
    parser.add_argument("--num_windows", type=int, default=10, help="Number of windows to extract per image")
    parser.add_argument("--output_folder", type=str, default="patches", help="Output folder for individual PNG files")
    parser.add_argument("--hdf5_chunks", type=int, default=128, help="Samples shape of chunks used when saving an HDF5 file")

    args = parser.parse_args()
    
    # Flipped due to cv2 conventions
    patch_width = args.patch_height
    patch_height = args.patch_width

    patch_size = (patch_width, patch_height)
    dataset_shape = (0, patch_width, patch_height, 1)
    maxdataset_shape = (None, patch_width, patch_height, 1)
    chunks_shape = (args.hdf5_chunks, patch_width, patch_height, 1)

    patch_stride = args.patch_stride
    num_windows = args.num_windows
    output_folder = args.output_folder
    output_count = 0

    output_folder = "{}-{}x{}-stride-{}".format(output_folder, patch_width, patch_height, patch_stride)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    hdf5_filename = HDF5_OUTPUT_PATTERN.format(patch_width, patch_height, args.patch_stride)
    hdf5_output = h5py.File(hdf5_filename, "w")
    out_dataset = hdf5_output.create_dataset("patches", shape=dataset_shape, maxshape=maxdataset_shape, chunks=chunks_shape, compression="gzip", compression_opts=7, dtype=np.uint8)
    out_maxindex = 0

    print("Saving to {}".format(hdf5_filename))

    for folder in IMAGE_FOLDERS:
        all_images = glob.glob("{}/*.png".format(folder))
        print("Folder {} found {} images".format(folder, len(all_images)))

        for idx, filename in enumerate(all_images):
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            h, w = image.shape
            mask = extract_polar_mask(image)
            windows = polar_sliding_window(image.shape, patch_size, mask, step_size=patch_stride)
            filtered_windows = filter_windows(windows, image, window_size=patch_size)
            print("({} / {}) - {} ({} by {}) - Generated {} windows, filtered to {}".format(idx, len(all_images), filename, w, h, len(windows), len(filtered_windows)))

            if len(filtered_windows) < num_windows:
                sampled_windows = filtered_windows
            else:
                sampled_windows = np.random.choice(filtered_windows, size=num_windows, replace=False)

            extracted_windows = extract_windows(image, sampled_windows)
            #painted_image = paint_windows(image, filtered_windows)

            for window_image in extracted_windows:
                out_filename = "{}/patch-{}.png".format(output_folder, output_count)
                cv2.imwrite(out_filename, window_image)

                if output_count >= out_maxindex:
                    out_maxindex = out_maxindex + args.hdf5_chunks
                    out_dataset.resize(out_maxindex, axis=0)

                out_dataset[output_count] = window_image.reshape((patch_width, patch_height, 1))
                output_count = output_count + 1

            del extracted_windows
            del image

        print("Total sampled patches {} HDF5 output length {}".format(output_count, out_dataset.shape[0]))
    if output_count < out_maxindex:
        out_dataset.resize(output_count - 1, axis=0)

        print("Resizing HDF5 dataset from {} to {}".format(out_maxindex, output_count))

    hdf5_output.close()
    print("Done")

    
