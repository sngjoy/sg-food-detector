import os
import re
import PIL
import pytest
import tensorflow as tf
import src.inference as inference

path = os.path.dirname(__file__)
img1 = os.path.join(path, 'kaya_toast.jpg')
img2 = os.path.join(path, 'kaya_toasted.jpg') # corrupted image
img3 = os.path.join(path, 'kaya_toast.wrong') # wrong image format
MODEL_DIR = "model/tensorfood.h5"

def test_image_integrity(img1=img1, img2=img2):
    assert inference._check_image_integrity(img1), 'Image is corrupted and unreadable.'
    with pytest.raises(PIL.UnidentifiedImageError):
        inference._check_image_integrity(img2)

def test_image_format(img1=img1, img3=img3):
    assert inference._check_image_format(img1), 'Image in wrong format.'
    with pytest.raises(TypeError):
        inference._check_image_format(img3)

def test_model_integrity(model=MODEL_DIR):
    model = tf.keras.models.load_model(model)
    img_arr = inference.load_image(img1)
    prob_list, _, _ = inference.make_prediction(model, img_arr)
    assert len(prob_list) == 12, 'Wrong number of classes, expected 12.'
    assert round(sum(prob_list), 5) == 1, 'Wrongly predicted probabilities, expected sum of probabilities to be 1.'

def test_image_arr(img1=img1):
    img_arr = inference.load_image(img1)
    assert img_arr.shape == (1, 224, 224, 3), 'Wrong image array shape, expected (1, 224, 224, 3).'


