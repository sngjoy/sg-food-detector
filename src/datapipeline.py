import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class Datapipeline:
    def __init__(
        self,
        data_path="../data.csv",
        data_aug={
            "rotation_range": None,
            "shear_range": None,
            "horizontal_flip": False,
            "zoom_range": 0.0,
        },
        target_size=(224, 224),
    ):

        self.df = pd.read_csv(data_path)
        self.rotation_range = data_aug["rotation_range"]
        self.shear_range = data_aug["shear_range"]
        self.horizontal_flip = data_aug["horizontal_flip"]
        self.zoom_range = data_aug["zoom_range"]
        self.target_size = target_size

    def load_generator(self):
        """Generate batches of tensor image data with real-time data augmentation."""
        self.df_train, self.df_test = train_test_split(
            self.df, test_size=0.2, stratify=self.df["labels"], random_state=18
        )
        self.train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            validation_split=0.2,
            rotation_range=self.rotation_range,
            shear_range=self.shear_range,
            horizontal_flip=self.horizontal_flip,
            zoom_range=self.zoom_range,
        )
        self.val_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            validation_split=0.25
        )
        self.test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator()

    def load_batches(self, image_col):
        """Generate batches of augmented data."""
        self.train_data_gen = self.train_image_generator.flow_from_dataframe(
            dataframe=self.df_train,
            x_col=image_col,
            y_col="labels",
            subset="training",
            shuffle=False,
            seed=18,
            target_size=self.target_size,
            batch_size=32,
        )
        self.val_data_gen = self.val_image_generator.flow_from_dataframe(
            dataframe=self.df_train,
            x_col=image_col,
            y_col="labels",
            subset="validation",
            shuffle=False,
            seed=18,
            target_size=self.target_size,
            batch_size=32,
        )
        self.test_data_gen = self.test_image_generator.flow_from_dataframe(
            dataframe=self.df_test,
            x_col=image_col,
            y_col="labels",
            shuffle=False,
            seed=18,
            target_size=self.target_size,
            batch_size=32,
        )
        return self.train_data_gen, self.val_data_gen, self.test_data_gen
