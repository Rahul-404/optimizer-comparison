import os
import tensorflow as tf
from models import create_model
from data_loader import load_dataset
import datetime

tf.config.run_functions_eagerly(True)
print("Eager execution:", tf.executing_eagerly())

def train_model(model_size, dataset_name, optimizer, epochs=5, batch_size=64):
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)
    input_shape = x_train.shape[1:]
    num_classes = 10

    model = create_model(model_size, input_shape, num_classes)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    log_dir = f"results/{dataset_name}/{model_size}/{type(optimizer).__name__}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )

    return history
