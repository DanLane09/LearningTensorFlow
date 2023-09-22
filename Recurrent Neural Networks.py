import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Multiple types of layers can be used here, with 'tanh' being the default activation: SimpleRNN, GRU, LSTM
# Can make the layers bidirectional by doing model.add(layers.Bidirectional(layers.LSTM(512, activation='tanh')))
# Using bidirectional layers doubles the number of nodes specified when adding the layer
model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(layers.LSTM(512, return_sequences=True, activation='tanh'))      # The default activation for RNNs is tanh
model.add(layers.LSTM(512, activation='tanh'))
model.add(layers.Dense(10))

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)
model.evaluate(x_test, y_test, batch_size=64, verbose=1)
