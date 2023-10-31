import asyncio
import websockets as ws
import numpy as np
import array
import json

import tensorflow as tf

# from nlp_communication import Communication

HAND_INDICES = list(range(0, 21))
POSE_INDICES = []

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

class Inferencer:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.signature_list = list(self.interpreter.get_signature_list().keys())

        if "serving_default" not in self.signature_list:
            raise ValueError(f"Signature 'serving_default' not found in provided model. Found {self.signature_list}")
        
        self.prediction_func = self.interpreter.get_signature_runner('serving_default')

    def predict(self, data):
        data = tf.cast(data, tf.float32)
        data = tf.reshape(data, (data.shape[0], data.shape[1], 1))


        prediction = self.prediction_func(conv1d_8_input=data)['dense_3']

        cats = np.zeros((1, 26))
        cats[np.arange(1), np.argmax(prediction)] = 1

        return cats
    
inferencer = Inferencer(model_path="/Users/jon/development/university/sis/models/cnn/model_2.tflite")

phrase = []

async def message_handler(websocket, path):
    global count
    count = 0
    async for msg in websocket:
        json_data = json.loads(msg)
        hand_landmarks = json_data['HandLandmarks'] if 'HandLandmarks' in json_data else None
        # pose_landmarks = json_data['PoseLandmarks'] if 'PoseLandmarks' in json_data else None
        if hand_landmarks is not None: #and pose_landmarks is not None:
            if count >= 10: # We want every 10th frame
                count = 0
            else:
                count += 1
                continue
            
            landmarks = []
            landmarks.extend(hand_landmarks[i]['X'] for i in HAND_INDICES)
            # landmarks.extend(pose_landmarks[i]['Y'] for i in POSE_INDICES)

            landmarks.extend(hand_landmarks[i]['Y'] for i in HAND_INDICES)
            # landmarks.extend(pose_landmarks[i]['Y'] for i in POSE_INDICES)

            landmarks.extend(hand_landmarks[i]['Z'] for i in HAND_INDICES)
            # landmarks.extend(pose_landmarks[i]['Z'] for i in POSE_INDICES)

            landmarks = np.array([landmarks])
            
            prediction_cats = inferencer.predict(landmarks)
            letter = ALPHABET[np.argmax(prediction_cats)] # Get the predicted letter

            # print(f"Predicted Letter: {letter}")
            if len(phrase) == 0 or phrase[-1] != letter:
                phrase.append(letter)
                print(phrase)

            if len(phrase) > 3:
                # XXX: Start sending to NLP for sentence building
                pass

        else: continue # Skip the loop if hand landmarks are not found


print("Starting server...")

start_server = ws.serve(message_handler, "127.0.0.1", 54174)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()