from model.model import ImageClassifier
from utils import *
from PIL import Image
import numpy as np
import flask
import io
from config.config import IMG_ROWS, IMG_COLS

# initialize our Flask application lication and the Keras model
application = flask.Flask(__name__)

cls = ImageClassifier()
model = cls.load_model()

# the code for the server goes here :)

@application.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    try:
    	# ensure an image was properly uploaded to our endpoint
	    if flask.request.method == "POST":
	        if flask.request.files.get("image"):
	            # read the image in PIL format
	            image = flask.request.files["image"].read()
	            image = Image.open(io.BytesIO(image))

	            # preprocess the image and prepare it for classification
	            image = prepare_image(image, target=(IMG_ROWS, IMG_COLS))

	            # classify the input image and then initialize the list
	            # of predictions to return to the client
	            preds = model.predict(image)
	            # print(preds)
	            # response = np.array_str(np.argmax(preds, axis=1))
                digit = np.argmax(preds)
                prediction = {'digit':int(digit)}

    except Exception as e:
    	raise e

    # return the data dictionary as a JSON response
    return flask.jsonify(prediction)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    application.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
