import base64
import os
import cv2 as cv
import numpy as np

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit

import redis

from redis_handler import RedisHandler
from socket_handler import SocketHandler

app = Flask(__name__)
app.config["SECRET"] = os.environ("API_SECRET")

socketio = SocketIO(app)

redisHandler = RedisHandler() # Create handler with default
socketHandler = SocketHandler(socketio, redisHandler)


def connect():
    print("Connected to websocket")
    emit("response", {"data": "connected", "status": 0})

uuid = None

@socketio.on("begin")
def received_begin():
    if socketHandler.transmission_in_progress:
        emit("response", {"data": "Transmission already in progress", "status": 1})
    else:
        uuid = socketHandler.create_transmission()
        emit("response", {"data": uuid, "status": 0})

@socketio.on("image")
def received_image(encoded_image):
    if uuid is None:
        emit("response", {"data": "No transmission in progress", "status": 1})
    else:
        uuid = socketHandler.handle_image(uuid, encoded_image)
        emit("response", {"transmission_id": uuid, "status": 0})

@socketio.on("end")
def received_end(uuid):
    if uuid is None:
        emit("response", {"data": "No transmission in progress", "status": 1})
    else:
        socketHandler.verify_transmission(uuid)
        emit("response", {"data": "Transmission ended", "status": 0})


