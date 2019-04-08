import math
import time

from autokeras import ImageClassifier
from keras.datasets import cifar10
from keras_preprocessing.image import ImageDataGenerator


def load_trained_model():
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        '/Users/michalstejskal/git/chobot-docs/train_files/flower_photos/flower_photos', target_size=(150, 150),
        batch_size=32, class_mode='binary')

    x_train = []
    y_train = []
    batch_index = 0
    while batch_index <= train_generator.n:
        # while batch_index <= train_generator.batch_index:
        print(str(batch_index) + '/' + str(train_generator.n))
        data = train_generator.next()
        x_train.append(data[0])
        y_train.append(data[1])
        batch_index = batch_index + 1

    print('processed ' + str(batch_index) + ' images. Start splitting.')
    train_test_split = math.ceil(batch_index * 0.75)
    x_test = x_train[train_test_split:]
    y_test = y_train[train_test_split:]
    x_train = x_train[:train_test_split]
    y_train = y_train[:train_test_split]

    print('Images splited into train and test dataset, start NAS training.')
    global clf
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=1 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    print('precission: ' + str(y * 100))


def predict(image):
    clf.predict(image)


load_trained_model()