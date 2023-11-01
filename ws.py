import asyncio
import websockets as ws
import numpy as np
import array
import json
import datetime, threading

import tensorflow as tf

from nlp_communication import Communication
from autocorrect import Speller

HAND_INDICES = list(range(0, 21))
POSE_INDICES = []

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

spell = Speller()

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
comms = Communication()


async def message_handler(websocket, path):
    global count, phrase, correctedIndex, lastFrameTime, newLetterCount
    count = 0
    correctedIndex = 0
    phrase = []
    lastFrameTime = None
    newLetterCount = 0
    async for msg in websocket:
        json_data = json.loads(msg)
        hand_landmarks = json_data['HandLandmarks'] if 'HandLandmarks' in json_data else None
        if hand_landmarks is not None: #and pose_landmarks is not None:

            currentTime = datetime.datetime.now()
            if lastFrameTime is not None:
                timeDiff = currentTime - lastFrameTime
                if timeDiff.seconds >= 1:
                    if len(phrase) > 0 and phrase[-1] not in "_ ":
                        print("Space added")
                        phrase.append('_')
                        await send_autocorrect(websocket, phrase)
                        count = 0

                if timeDiff.seconds >= 10:
                    count = 0
                    phrase = []
                    await send_autocorrect(websocket, phrase)
                    comms.reset_string()
                    lastFrameTime = None

            lastFrameTime = currentTime
            if count >= 30: # We want every 10th frame
                count = 0
            else:
                print(f"Increasing count: {count + 1}")
                count += 1
                continue
            
            print('Captured frame')

            landmarks = []
            landmarks.extend(hand_landmarks[i]['X'] for i in HAND_INDICES)
            landmarks.extend(hand_landmarks[i]['Y'] for i in HAND_INDICES)
            landmarks.extend(hand_landmarks[i]['Z'] for i in HAND_INDICES)

            landmarks = np.array([landmarks])
            
            prediction_cats = inferencer.predict(landmarks) # Get prediction
            letter = ALPHABET[np.argmax(prediction_cats)] # Get the predicted letter

            if letter == 'u': # Fixes mistake with ML recognition
                letter = 'r'
            
            phrase.append(letter)
            print(phrase)
            newLetterCount += 1

            # if len(phrase) == 5:
            #     phrase.append(' ')

            # if len(phrase) > 2:
            #     await send_autocorrect(websocket, phrase)
            # else:
            #     await send_autocorrect(websocket, phrase)

            if newLetterCount >= 3:
                print("Calling OpenAI")
                result = comms.new_letter(letter)
                newLetterCount = 0
                # print(f"Result: {result}")

                if result is not None:
                    corrected_result = ''.join(result)
                    print(f"Corrected Result: {corrected_result}")
                    # phrase = [*corrected_result]

                    print(f"New phrase: {phrase}")

                    comms.reset_string()
                    comms.new_letter_no_conversion(phrase)

                    await websocket.send(corrected_result)
                else:
                    await websocket.send(''.join(phrase))
            else:
                comms.new_letter_no_conversion(letter)
                await websocket.send(''.join(phrase))



        else: 
            continue # Skip the loop if hand landmarks are not found

async def send_autocorrect(websocket, phrase):
    string = ''.join(phrase)
    string = string.replace('_', ' ')
    await websocket.send(spell(string))

print("Starting server...")

start_server = ws.serve(message_handler, "127.0.0.1", 54174)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()