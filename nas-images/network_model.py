import time

from autokeras import ImageClassifier
from keras.datasets import cifar10
from keras_preprocessing.image import ImageDataGenerator


def load_trained_model():
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        '/Users/michalstejskal/git/chobot-docs/train_files/flower_photos/flower_photos', target_size=(150, 150),
        batch_size=32, class_mode='binary')

    train_test_split = train_generator.samples * 0.75

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    idx = 0
    for item in train_generator:
        print('process item ' + str(idx))
        if idx < train_test_split:
            x_train.append(item[0])
            y_train.append(item[1])
        else:
            x_test.append(item[0])
            y_test.append(item[1])
        idx += 1

    global clf
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=1 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    print('precission: ' + str(y * 100))


def predict(image):
    clf.predict(image)


load_trained_model()
