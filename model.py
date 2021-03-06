import csv
import cv2
import numpy as np

# I used my own data as well as data given by Udacity
data_source = ['./datas/20170516/', './datas/data/']
lines = []

# load log content
for i in range(len(data_source)):
    lines.append([])
    with open(data_source[i] + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines[i].append(line)

print("Loading images..")

images = []
measurements = []

# helper method to clean up the code
def addImage(filepath, measurement, correction=0.0):
    global images, measurements
    image = cv2.imread(filepath)
    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    measurements.append(measurement+correction)
    
# for all data sources, load data
for i in range(len(data_source)):
    for line in lines[i]:
        correction = 0.2
        source_path = line[0]
        source_path_left = line[1]
        source_path_right = line[2]

        filepath = data_source[i] + 'IMG/' + source_path.split("/")[-1]
        addImage(filepath, measurement=float(line[3]))

        # if given image is steering right, add image from left side with correction
        if (float(line[3]) > 0):
            filepath = data_source[i] + 'IMG/' + source_path_left.split("/")[-1]
            addImage(filepath, measurement=float(line[3]),correction=correction)

        # if given image is steering left, add image from right side with correction
        if (float(line[3]) < 0):
            filepath = data_source[i] + 'IMG/' + source_path_right.split("/")[-1]
            addImage(filepath, measurement=float(line[3]), correction=correction*-1.0)

    augmented_images, augmented_measurements = [], []
    # flip images with steering right/left and add to data
    for image,measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        if (i==0 and abs(measurement) > 0.2):
            augmented_images.append(cv2.flip(image,1))
            augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Cropping2D
from keras.layers import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: K.tf.image.resize_images(x, (80, 160)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, ))
model.add(Cropping2D(cropping=((35,12), (0,0)), input_shape=(80,160,3)))
model.add(GaussianNoise(0.2))
model.add(Conv2D(24,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(48,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(64,(5,5),activation="relu"))
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(50,activation="relu"))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')