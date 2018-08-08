# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "5.png"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

try:
	# submit the request
	r = requests.post(KERAS_REST_API_URL, files=payload).json()
	print r

except Exception as e:
	raise e

# To call from terminal
#curl -X POST -F image=@5.png 'http://localhost:5000/predict'