'''
https://www.tensorflow.org/tutorials/quickstart/beginner
'''

import tensorflow as tf


# Load a dataset

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a machine learning model

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),  # relu sigmoid
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print(predictions)

print(tf.nn.softmax(predictions).numpy())

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',  loss=loss_fn, metrics=['accuracy'])

# Train and evaluate your model

model.fit(x_train, y_train, epochs=2)

print(model.evaluate(x_test,  y_test, verbose=2))  #


# If you want your model to return a probability, you can wrap the trained model, and attach the softmax
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
print()
print(y_test[1])  # second photo
print(probability_model(x_test[1:2]))  # https://www.w3schools.com/python/numpy/numpy_array_slicing.asp











