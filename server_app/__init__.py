from flask import Flask, request
from .setup import setup_model
import numpy as np
from PIL import Image

app = Flask(__name__)
model = setup_model()

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    file = request.files.get("image", None)

    if file is None:
        return "Incorrect data"
    
    image = np.load(file.stream)
    Image.fromarray(image).save("bakeit.jpg")
    Image.fromarray(image).save("bakeit.png")

    return image