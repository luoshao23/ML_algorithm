
# coding: utf-8


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from SR_module import data_prop, label_onehot_encoder

import os
import pandas as pd
import numpy as np
import argparse




def load_img(img_list):

    imgs = []
    for img_path in img_list:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        imgs.append(x)

    imgs = np.concatenate(imgs, 0)
    imgs = preprocess_input(imgs)
    return imgs


# ## load images and labels


def BelleNet(input_shape=(224, 224, 3), labels=29):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    # !IMPORTANT: use sigmoid in multi-label task
    model.add(Dense(labels, activation='sigmoid'))

    return model

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


def main():
    parser = argparse.ArgumentParser(description='BelleNet')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='Epochs of the training')

    args = parser.parse_args()

    data_root = '/home/luo.sz/python/data/'
    img_root = '/home/luo.sz/python/data/imgs/'

    properties = data_prop(
        os.path.join(data_root, '16spring_m.csv'))
    properties17 = data_prop(
        os.path.join(data_root, '17spring_m.csv'))

    p_total = pd.concat([properties, properties17])[
        [u'产品系列(旧)', u'款型', u'跟型', u'跟高', u'帮面颜色', u'有无配饰']]

    base_list = os.listdir(img_root)

    file_list_2016 = ['%s%s.jpg' % (
        img_root, f) for f in properties.index if '%s.jpg' % f in base_list]
    file_list_2017 = ['%s%s.jpg' % (
        img_root, f) for f in properties17.index if '%s.jpg' % f in base_list]

    img16 = load_img(file_list_2016)
    img17 = load_img(file_list_2017)

    label16 = p_total.loc[
        [img.split('/')[-1].split('.')[0] for img in file_list_2016]]
    label17 = p_total.loc[
        [img.split('/')[-1].split('.')[0] for img in file_list_2017]]
    label_all = pd.concat([label16, label17])

    lohe = label_onehot_encoder()
    lohe.fit(label_all)
    label_one16 = lohe.transform(label16)
    label_one17 = lohe.transform(label17)

    X, y = np.concatenate([img16, img17], axis=0), np.concatenate(
        [label_one16.toarray(), label_one17.toarray()], axis=0)

    total_len = y.shape[0]
    train_len = int(total_len * 0.7)

    np.random.seed(23)
    random_index = np.random.permutation(y.shape[0])
    train_index, test_index = random_index[
        :train_len], random_index[train_len:]
    X_train, y_train = X[train_index, :], y[train_index, :]
    X_test, y_test = X[test_index, :], y[test_index, :]

    epochs = args.epochs

    model = BelleNet()
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])  # !IMPORTANT: use binary_crossentropy in multi-label task

    train_datagen = image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

    test_datagen = image.ImageDataGenerator(rescale=1. / 255)

    model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=32),
                        steps_per_epoch=len(X_train) / 32, epochs=epochs)

    model.evaluate_generator(test_datagen.flow(X_test, y_test))

    model.save_weights(os.path.join(data_root, 'BelleNet_Weights.h5'))
    json_string = model.to_json()
    with open(os.path.join(data_root, 'BelleNet.json'), 'w') as f:
        f.write(json_string)


if __name__ == '__main__':
    main()