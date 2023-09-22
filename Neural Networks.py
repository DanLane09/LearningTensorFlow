import os
from tensorflow import keras
from keras import layers
from keras.datasets import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0  # We use 784 as that is 28^2 (size of data set)
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0  # We divide by 255 to normalise the numbers

# Sequential API  (Very convenient, not very flexible)
# These next steps can also be added one by one using model.add()   This can be useful for debugging
model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),
        # You are able to name layers to distinguish them more easily (see below)
        layers.Dense(512, activation="relu", name="first_layer"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ]
)

# Getting information about individual layers
# This can be done with any API and can be useful for debugging
'''
model = keras.Model(inputs=model.inputs,
                    outputs=[model.layers[-2].output])
feature = model.predict(x_train)
print(feature.shape)
'''
'''
---This is showing how to look at individual layers---
model = keras.Model(inputs=model.inputs,
                    outputs=[model.get_layer("first_layer").output])

---This can also be done for individual layers one at a time---
model = keras.Model(inputs=model.inputs,
                    outputs=[layer.output for layer in model.layers])
features = model.predict(x_train)
for feature in features:
    print(feature.shape)
'''

# Functional API  (More flexible)
# This is another way of adding layers to a model
'''
inputs = keras.Input(shape=(28 * 28))
x = layers.Dense(2048, activation="relu")(inputs)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dense(512, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
'''

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# batch_size is the number of elements from the training data used in each epoch
# epochs is the number of times the model is run on the training data
# verbose is how the progress is displayed to the user: 0 = nothing, 1 = progress bar, 2 = summary of each epoch
model.fit(x_train, y_train, batch_size=32, epochs=8, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
