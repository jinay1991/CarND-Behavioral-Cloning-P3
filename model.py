import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv
import logging
from keras.utils import plot_model


logging.basicConfig(level=logging.INFO)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            correct_factor = [0.2, 0.0, -0.2]
            for batch_sample in batch_samples:
                for i in range(3):
                    # name = batch_sample[i]
                    name = os.path.join(os.path.dirname(__file__), "data", "IMG", batch_sample[0].split("/")[-1])
                    image = cv2.imread(name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if image is None:
                        logging.error("Failed to open %s" % (name))
                        continue
                    angle = float(batch_sample[3]) + correct_factor[i]
                    images.append(image)
                    angles.append(angle)
                    images.append(cv2.flip(image, 1))
                    angles.append(angle * -1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    args = parser.parse_args()

    lines = []
    with open(os.path.join(args.input, "data", "driving_log.csv")) as fp:
        reader = csv.reader(fp)
        for line in reader:
            lines.append(line)

    logging.info("processing each image for %s" % (len(lines)))

    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    train_generator = generator(train_samples, 32)
    validation_generator = generator(validation_samples, 32)

    # --- NVIDIA Model ---
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    plot_model(model, to_file='model.png')

    model.compile(optimizer='adam', loss='mse')
    model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
                        validation_data=validation_generator, validation_steps=len(validation_samples), epochs=1, verbose=1)

    model.save('model.h5')
