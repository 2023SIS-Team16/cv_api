import base64
import os
import cv2 as cv
import numpy as np

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit

import redis

from redis.connection_handler import RedisConnectionDispatcher
# from redis.redis_connection import RedisConnection
# from socket_handler import SocketHandler

app = Flask(__name__)
app.config["SECRET"] = os.environ.get("API_SECRET")

socketio = SocketIO(app)

dispatcher = RedisConnectionDispatcher() # Create handler with default
# socketHandler = SocketHandler(socketio, redisHandler)


@socketio.on("connect")
def connect():
    print("Connected to websocket")
    emit("response", {"data": "connected", "status": 0})

@socketio.on("start_trans")
def start_transmission(identifier):
    connection = dispatcher.get_connection()
    res = connection.check_transmission_exists(identifier)

    if res:
        emit("response", {"error": "Already exists", "status": 406})
    else: # Transmission does not exist
        connection.create_transmission(identifier)
        emit("response", {"data": "Transmission created", "status": 200})

@socketio.on("submit_frame")
def submit_frame_for_transmission(identifier, image):
    connection = dispatcher.get_connection()
    res = connection.check_transmission_exists(identifier)

    if res:
        connection.set_transmission_block(identifier, image)
        emit("response", {"data": "Frame submitted", "status": 200})
    else:
        emit("response", {"error": "Transmission does not exist", "status": 404})


