import cv2
import random

import numpy as np
import tensorflow as tf

class ImageDatasetFromImagePathsAndLabelTF(tf.keras.utils.Sequence):
    def __init__(self, imagePaths, labels, image_transform, batch_size, shuffle = True):
        self.labels = labels
        self.imagePaths = imagePaths
        self.ids = list(range(len(imagePaths)))

        self.image_transform = image_transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.on_epoch_end()
        
    def read_image(self, image_path):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = self.image_transform(image = image)["image"]
        return image

    def __len__(self):
        return len(self.imagePaths) // self.batch_size + int(len(self.imagePaths) % self.batch_size != 0)

    def on_epoch_end(self):
        if self.shuffle:
            self.ids = sorted(self.ids, key=lambda k: random.random())

    def __getitem__(self, index):
        ids = self.ids[index*self.batch_size: min((index+1)*self.batch_size, len(self.ids))]
        images = np.array([self.read_image(self.imagePaths[i]) for i in ids])
        label = np.array([self.labels[i] for i in ids])

        return {"image": images}, {"output": label}