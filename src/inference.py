import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow warning logs
import sys
import PIL
import numpy as np
from PIL import Image
import tensorflow as tf

def _check_image_integrity(img_dir):
    try:
        PIL.Image.open(img_dir)
        return True
    except PIL.UnidentifiedImageError as error:
        raise error
    
def _check_image_format(img_dir):
    if img_dir.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.gif')):
        return True
    raise TypeError(f"Image format is not accepted.")


def load_image(img_dir):
    img = PIL.Image.open(img_dir).convert('RGB')
    img = img.resize(size=(224, 224))
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

def make_prediction(model, img_arr):
    FOODS = ['chilli_crab',
            'curry_puff',
            'dim_sum',
            'ice_kacang',
            'kaya_toast',
            'nasi_ayam',
            'popiah',
            'roti_prata',
            'sambal_stingray',
            'satay',
            'tau_huay',
            'wanton_noodle']
    prob_list = model.predict(img_arr).squeeze()
    idx = prob_list.argmax()
    pred = FOODS[idx]
    prob = prob_list[idx]
    return prob_list, pred, prob

if __name__ == '__main__':
    MODEL_DIR = 'tensorfood.h5'
    img_dir = sys.argv[1]
    _check_image_integrity(img_dir)
    _check_image_format(img_dir)
    model = tf.keras.models.load_model(MODEL_DIR)
    img_arr = load_image(img_dir)
    _, pred, prob = make_prediction(model, img_arr)
    print(f'Predicted {pred} with {round(prob,3)} probability')