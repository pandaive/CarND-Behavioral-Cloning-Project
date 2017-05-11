import csv
import cv2
import numpy as np

lines = []
with open('./datas/20170505/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []



for line in lines:
    correction = 0.2
    source_path = line[0]
    source_path_left = line[1]
    source_path_right = line[2]

    filename = source_path.split("/")[-1]
    filepath = './datas/20170505/IMG/' + filename
    image = cv2.imread(filepath)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

    filename = source_path_left.split("/")[-1]
    filepath = './datas/20170505/IMG/' + filename
    image = cv2.imread(filepath)
    images.append(image)
    measurement = float(line[3]) + correction
    measurements.append(measurement)

    filename = source_path_right.split("/")[-1]
    filepath = './datas/20170505/IMG/' + filename
    image = cv2.imread(filepath)
    images.append(image)
    measurement = float(line[3]) - correction
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []

for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)


#from sklearn.utils import shuffle
#images, measurements = shuffle(images, measurements)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
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

model.save('model_20170511.h5')