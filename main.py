import cnn2d
import dataload
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

physical_device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)

EmoDB_file_path = '/home/hby/Documents/DataSet/EmoDB'


def train(train_data_x, train_data_y, validation_data_x, validation_data_y):
    model = cnn2d.model2d(input_shape=(128, 251, 1), num_classes=7)
    model.summary()
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=0,
                       patience=20)

    mc = ModelCheckpoint('model.h5',
                         monitor='val_categorical_accuracy',
                         mode='max',
                         verbose=0,
                         save_best_only=True)

    model.fit(train_data_x, train_data_y,
              validation_data=(validation_data_x, validation_data_y),
              epochs=100,
              batch_size=4,
              verbose=2,
              callbacks=[es, mc])


def test(test_data_x, test_data_y ):
    new_model = load_model('model.h5')
    new_model.evaluate(test_data_x, test_data_y, batch_size=1)


if __name__ == '__main__':

    train_data_x, train_data_y, validation_data_x, validation_data_y, test_data_x, test_data_y = dataload.load_data(EmoDB_file_path)

    train_data_x = normalize(train_data_x)
    validation_data_x = normalize(validation_data_x)
    test_data_x = normalize(test_data_x)

    train_data_y = to_categorical(train_data_y)
    validation_data_y = to_categorical(validation_data_y)
    test_data_y = to_categorical(test_data_y)

    train(train_data_x, train_data_y, validation_data_x, validation_data_y)

    test(test_data_x, test_data_y)
