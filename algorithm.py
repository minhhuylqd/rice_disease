import numpy as np
from os import listdir
from keras.models import load_model
from PIL import Image

def create_model():
	model = load_model('RiceModel.h5')
	return model