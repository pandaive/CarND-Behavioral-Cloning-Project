import csv
import cv2
import numpy as np
import sklearn

lines = []
with open('./datas/20170516/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def addImage(filepath, measurement, correction=0.0):
    image = cv2.imread(filepath)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), measurement+correction
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                source_path_left = batch_sample[1]
                source_path_right = batch_sample[2]

                filepath = './datas/20170516/IMG/' + source_path.split("/")[-1]
                image, measurement = addImage(filepath, measurement=float(batch_sample[3]))
                images.append(image)
                measurements.append(measurement)

                if (float(batch_sample[3]) >= 0):
                    filepath = './datas/20170516/IMG/' + source_path_left.split("/")[-1]
                    image, measurement = addImage(filepath, measurement=float(batch_sample[3]),correction=correction)
                    images.append(image)
                    measurements.append(measurement)

                if (float(batch_sample[3]) <= 0):
                    filepath = './datas/20170516/IMG/' + source_path_right.split("/")[-1]
                    image, measurement = addImage(filepath, measurement=float(batch_sample[3]), correction=correction*-1.0)
                    images.append(image)
                    measurements.append(measurement)

            augmented_images, augmented_measurements = [], []
            for image,measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

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
model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model_20170515_gen.h5')