# from hand_segment import HandSegmenter

import base64
import os
import cv2 as cv #pip install pip install opencv-python
import numpy as np
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit

# segmenter = HandSegmenter(maxHands=1, boundingBoxOffset=(20, 20))

app = Flask(__name__)
app.config["SECRET"] = "obivously not a secret"

socketio = SocketIO(app)

savePath = os.path.join(os.getcwd(), "img.jpg")

def convert_encoded_image(encoded):
    data = encoded.split(',')[1]
    bytes = base64.b64decode(data)
    array = np.frombuffer(bytes, dtype=np.uint8)
    image = cv.imdecode(array, cv.IMREAD_COLOR)
    return image

@socketio.on("connect")
def connect():
    print("Connected")
    emit("response", {"data": "connected"})

@socketio.on("image")
def received_image(encoded_image):
    image = convert_encoded_image(encoded_image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite(savePath, gray)
    emit("response", {"data": "image received"})


if __name__ == "__main__":
    socketio.run(app, debug=True, port=8765, host='0.0.0.0')