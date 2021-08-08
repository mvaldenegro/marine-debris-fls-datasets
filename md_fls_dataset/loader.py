import numpy as np
import cv2

import os
import glob
import pathlib

OBJECT_CLASES = [
    "bottle",
    "can",
    "carton",
    "box",
    "bidon",
    "pipe",
    "platform",
    "propeller",
    "sachet",
    "tire",
    "valve",
    "wrench"
]

MATERIAL_CLASSES = [
    "plastic",
    "metal",
    "rubber",
    "glass",
    "cardboard",
    "platform"
]

MATERIAL_OBJECT_MAPPING = {
    "plastic": ["drink-sachet", "plastic-bidon", "plastic-bottle", "plastic-pipe", "plastic-propeller"],
    "metal": ["can", "metal-bottle", "metal-box", "valve", "wrench"],
    "rubber": ["large-tire", "small-tire"],
    "glass": ["brown-glass-bottle", "glass-bottle", "glass-jar", "potion-glass-bottle"],
    "cardboard": ["drink-carton"],
    "platform": ["rotating-platform"]
}

OBJECT_CLASS_MAPPING = {
    "bottle": ["plastic-bottle", "metal-bottle", "glass-bottle", "potion-glass-bottle"],
    "can": ["can"],
    "carton": ["drink-carton"],
    "box": ["metal-box"],
    "bidon": ["plastic-bidon"],
    "pipe": ["plastic-pipe"],
    "platform": ["rotating-platform"],
    "propeller": ["plastic-propeller"],
    "sachet": ["drink-sachet"],
    "tire": ["large-tire", "small-tire"],
    "valve": ["valve"],
    "wrench": ["wrench"]
}

INDIVIDUAL_OBJECT_CLASSES = [
    "brown-glass-bottle",  "drink-carton", "glass-bottle", "large-tire", "metal-box", "plastic-bottle", "plastic-propeller", "rotating-platform", "valve", "can",
    "drink-sachet", "glass-jar", "metal-bottle", "plastic-bidon", "plastic-pipe", "potion-glass-bottle", "small-tire", "wrench"
]

TURNTABLE_DATA_FOLDER = os.path.join(pathlib.Path(__file__).parent.absolute() ,"data/turntable-cropped")

VALID_DATASETS = ["turntable"]
VALID_CROPS = ["platform", "object"]
VALID_TASKS = ["object_classes", "individual_objects", "materials"]

SPLIT_SEED = 42

def load_debris_dataset(dataset="turntable", task="object_classes", crop="platform", image_size = (96, 96), split=None, add_channels_dim=True):
    """
        Loads the dataset from disk and returns all images in the specified task.

        Valid tasks are:
          - object_classes Classes correspond to the high level objects like can, bottle, tire, valve, wrench, etc.
          - individual_objects Individual objects are each its own class (18 in total).
          - materials Classes correspond to the physical materials in the objects (like plastic, metal, rubber, glass, cardboard)

        Valid crops can be:
          - platform Use image crops containing both the rotating platform and the object.
          - object Use image crops only containing the object.
    """

    assert dataset in VALID_DATASETS
    assert task in VALID_TASKS
    assert crop in VALID_CROPS

    if task == "object_classes":
        object_images, object_labels = load_turntable_objects(OBJECT_CLASS_MAPPING, OBJECT_CLASES, image_size, crop)

        return make_split(object_images, object_labels, split)

    if task == "individual_objects":
        object_images, object_labels = load_turntable_objects({x: x for x in INDIVIDUAL_OBJECT_CLASSES},
                                                              INDIVIDUAL_OBJECT_CLASSES, image_size, crop)

        return make_split(object_images, object_labels, split)

    if task == "materials":
        object_images, object_labels = load_turntable_objects(MATERIAL_OBJECT_MAPPING, MATERIAL_CLASSES, image_size, crop)

        return make_split(object_images, object_labels, split)

    raise NotImplementedError()

def class_names(dataset, task):
    pass

def make_split(object_images,  object_labels, split_type, random_seed=None, add_channels_dim=True):
    if split_type is None:
        x_all = np.concatenate(object_images, axis=0)
        y_all = np.concatenate(object_labels, axis=0)

        if add_channels_dim:
            x_all = np.expand_dims(x_all, axis=3)

        return x_all, y_all

    if split_type == "70/30":
        from sklearn.model_selection import train_test_split

        x_all = np.concatenate(object_images, axis=0)
        y_all = np.concatenate(object_labels, axis=0)

        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, shuffle=True, test_size=0.3, random_state=SPLIT_SEED)

        if add_channels_dim:
            x_train = np.expand_dims(x_train, axis=3)
            x_test = np.expand_dims(x_test, axis=3)

        return x_train, y_train, x_test, y_test

    #if split_type == "sequence":
        

def load_turntable_objects(dataset_map, class_names, size, crop):
    object_images, object_labels = [], []

    for class_name, objects in dataset_map.items():
        for object_name in objects:
            obj_images = load_turntable_individual_object(object_name, size, crop)
            label = class_names.index(class_name)

            for obj in obj_images:
                object_images.append(obj)
                object_labels.append([label] * len(obj))

    return object_images, object_labels

def load_turntable_individual_object(object_name, size, crop="platform"):
    """
        Loads all images for a particular object in the dataset.
    """
    
    assert crop in VALID_CROPS

    standing_glob_pattern = "{}/{}/{}-standing-frame-*.png".format(TURNTABLE_DATA_FOLDER, object_name, crop)
    sideways_glob_pattern = "{}/{}/{}-sideways-frame-*.png".format(TURNTABLE_DATA_FOLDER, object_name, crop)

    standing_files = glob.glob(standing_glob_pattern)
    sideways_files = glob.glob(sideways_glob_pattern)

    standing_num_files = len(standing_files)
    sideways_num_files = len(sideways_files)

    standing_file_pattern = "{}/{}/{}-standing-frame-{:03}.png"
    sideways_file_pattern = "{}/{}/{}-sideways-frame-{:03}.png"

    standing_images = []
    sideways_images = []

    for i in range(standing_num_files):
        filepath = standing_file_pattern.format(TURNTABLE_DATA_FOLDER, object_name, crop, i)
        image = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_LINEAR)

        standing_images.append(image)

    for i in range(sideways_num_files):
        filepath = sideways_file_pattern.format(TURNTABLE_DATA_FOLDER, object_name, crop, i)
        image = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_LINEAR)

        sideways_images.append(image)

    if standing_num_files > 0 and sideways_num_files > 0:
        return [np.array(standing_images), np.array(sideways_images)]

    if standing_num_files > 0 and sideways_num_files == 0:
        return [np.array(standing_images)]

    if sideways_num_files > 0 and standing_num_files == 0:
        return [np.array(sideways_images)]

    raise ValueError("No images found")