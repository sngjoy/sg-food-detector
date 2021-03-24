import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class Model:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def build_model(self):
        img_shape = self.target_size + (3,)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=img_shape, include_top=False, weights="imagenet"
        )
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
        dense_1 = tf.keras.layers.Dense(512, activation="relu")
        dense_2 = tf.keras.layers.Dense(128, activation="relu")
        dense_3 = tf.keras.layers.Dense(32, activation="relu")
        dense_4 = tf.keras.layers.Dense(16, activation="relu")
        prediction_layer = tf.keras.layers.Dense(12, activation="softmax")

        inputs = tf.keras.Input(shape=img_shape)
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = global_avg_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = dense_1(x)
        x = dense_2(x)
        x = dense_3(x)
        x = dense_4(x)
        outputs = prediction_layer(x)
        self.model = tf.keras.Model(inputs, outputs)

        learning_rate = 0.0001
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return self.model.summary()

    def fit(self, train_data_gen, val_data_gen, model_output_path, epochs=100):
        """Model fitting with train and val data"""
        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            verbose=1,
            patience=15,
            min_delta=0.00001,
        )

        checkpoint = ModelCheckpoint(
            filepath=model_output_path,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max",
        )

        self.model.fit_generator(
            generator=train_data_gen,
            validation_data=val_data_gen,
            steps_per_epoch=len(train_data_gen),
            epochs=epochs,
            callbacks=[early_stopping, checkpoint],
        )

    def evaluate(self, test_data_gen, model=None):
        """Computing the loss and accuracy of the model using test data
        Args:
            test_data_gen (DataFrameIterator): test data generated in batches
        Returns:
            loss, accuracy
        """
        self.test_data_gen = test_data_gen
        if model is None:
            model = self.model
        loss, accuracy = model.evaluate_generator(
            generator=test_data_gen,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            verbose=0,
        )
        return loss, accuracy

    def error_evalutaion(self):
        """Identify wrongly classified images in test set and save their names as a csv file"""
        pred = self.model.predict_generator(self.test_data_gen)
        pred_index = np.argmax(pred, axis=1)
        fnames = self.test_data_gen.filenames
        errors = np.where(pred_index != self.test_data_gen.classes)[0]
        wrongly_classified_images = []
        for i in errors:
            wrongly_classified_images.append(fnames[i])
        df = pd.DataFrame(data=wrongly_classified_images)
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        df.to_csv(
            f"/wrongly_classified_images/wrongly_classified_images_{self.timestr}.csv",
            index=False,
            header=False,
        )

    def save_model(self, path=model / model.h5):
        """Save model in Polyaxon's persistence storage"""
        self.model.save(path)
