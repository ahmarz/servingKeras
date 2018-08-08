from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import io
import cv2
from skimage import data

def prepare_image(image, target):
	"""
    prepares an image in the correct input for the keras model
    :Args:
    	image : PIL image send by curl
    	target: image resize target
    :return: the trained saved model
    """
	image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image,target)
	#image /=255
	image = image.reshape(1, 28, 28, 1)
	return image