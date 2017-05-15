import csv
import cv2
import numpy as np

lines = []
with open('./datas/20170516/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print("Loading images..")

images = []
measurements = []

def addImage(filepath, measurement, correction=0.0):
    global images, measurements
    image = cv2.imread(filepath)
    #image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    measurements.append(measurement+correction)
    

for line in lines:
    correction = 0.2
    source_path = line[0]
    source_path_left = line[1]
    source_path_right = line[2]

    filepath = './datas/20170516/IMG/' + source_path.split("/")[-1]
    addImage(filepath, measurement=float(line[3]))

    if (float(line[3]) > 0):
        filepath = './datas/20170516/IMG/' + source_path_left.split("/")[-1]
        addImage(filepath, measurement=float(line[3]),correction=correction)

    if (float(line[3]) < 0):
        filepath = './datas/20170516/IMG/' + source_path_right.split("/")[-1]
        addImage(filepath, measurement=float(line[3]), correction=correction*-1.0)

augmented_images, augmented_measurements = [], []

for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: K.tf.image.resize_images(x, (80, 160)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, ))
model.add(Cropping2D(cropping=((35,12), (0,0)), input_shape=(80,160,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model_20170515.h5')