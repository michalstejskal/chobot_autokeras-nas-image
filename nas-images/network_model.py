import csv
import os
import numpy as np
import tensorflow as tf

from configuration.app_config import model_data_path
from bo.network_dao import add_network_parameter
from bo.models import NetworkParameter
from autokeras import ImageClassifier
from autokeras.image.image_supervised import load_image_dataset
from autokeras.utils import pickle_from_file
from shutil import copyfile, rmtree


def prepare_train_directory(train_zip_dir):
    print('training model started')
    custom_path = train_zip_dir.replace('.zip', '')
    custom_path = train_zip_dir[0:train_zip_dir.rfind('/')]
    zip_ref = train_zip_dir.ZipFile(train_zip_dir, 'r')
    zip_ref.extractall(custom_path)
    zip_ref.close()
    return train_zip_dir.replace('.zip', '')


def prepare_labels(train_path):
    class_dirs = [i for i in os.listdir(path=train_path) if os.path.isdir(os.path.join(train_path, i))]

    global label_dict
    label_dict = {}
    label = 0
    for current_class in class_dirs:
        label_dict[str(label)] = current_class
        label += 1


def prepare_train_images(train_dir):
    if os.path.exists('/tmp/train/'):
        rmtree('/tmp/train/', ignore_errors=True)
    os.makedirs('/tmp/train/')

    class_dirs = [i for i in os.listdir(path=train_dir) if os.path.isdir(os.path.join(train_dir, i))]
    with open('/tmp/label.csv', 'w') as train_csv:
        fieldnames = ['File Name', 'Label']
        writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
        writer.writeheader()
        label = 0
        for current_class in class_dirs:
            for image in os.listdir(os.path.join(train_dir, current_class)):
                writer.writerow({'File Name': str(image), 'Label': label})
                copyfile(train_dir + '/' + str(current_class) + '/' + str(image), '/tmp/train/' + str(image))
                print('proccess image ' + str(image) + ' with label ' + str(current_class) + ' to file ' + '/tmp/train/' + str(image))
            label += 1
        train_csv.close()
        return '/tmp/label.csv'


def train_model(x_train, y_train):
    global clf
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=3 * 60 * 60)
    print('NAS end, start final fit')
    clf.final_fit(x_train, y_train, x_train, y_train, retrain=True)

    print('final fit end')
    y = clf.evaluate(x_train, y_train)
    print('precision: ' + str(y * 100))

    # save model
    print('export model')
    model_file_name = model_data_path + '/autokeras_model.pkl'
    clf.export_autokeras_model(model_file_name)


def load_trained_model(network):
    train_path = ""
    trained = False

    for parameter in network.parameters:
        if parameter.abbreviation == "IS_TRAINED" and parameter.value is not None and parameter.value.lower() == 'true':
            trained = True

        if parameter.abbreviation == "TRAIN_DATA_PATH":
            train_path = parameter.value

    if trained is False:
        labels_path = prepare_train_images(train_path)
        train_data_path = '/tmp/train/'
        x_train, y_train = load_image_dataset(csv_file_path=labels_path, images_path=train_data_path)
        train_model(x_train, y_train)

        parameter = NetworkParameter('IS_TRAINED', 'IS_TRAINED', True, network.network_id)
        add_network_parameter(parameter)

    # load labels and model
    prepare_labels(train_path)
    print('load model')
    model_file_name = model_data_path + '/autokeras_model.pkl'
    global model
    model = pickle_from_file(model_file_name)


def prepare_image_for_predict(image, target, input_mean=0, input_std=255):
    float_caster = tf.cast(image, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [target[0], target[1]])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def predict(image, additional_info):
    image = prepare_image_for_predict(image, target=(224, 224), input_mean=128, input_std=128)
    preds = model.predict(image)
    data = {"predictions": [label_dict[preds[0]]]}
    return data
