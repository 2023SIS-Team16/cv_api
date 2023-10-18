from inference_testing.landmarks import LandmarkProcessor
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2 as cv

import logging
import os
import json
import uuid

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

class DatasetBuilder:
    def __init__(self, frame_paths, output_path, classes, landmark_processor: LandmarkProcessor):
        self.frame_paths = frame_paths
        self.classes = classes
        self.landmark_processor = landmark_processor
        self.output_path = output_path

    def build_train(self, collected_landmarks, entries_per_rec=300):
        assert collected_landmarks != None, "collected_landmarks cannot be None"
        assert len(collected_landmarks) == 3, "collected_landmarks must have a size of 3 (pose, right_hand, face)"
        assert len(collected_landmarks[1]) > 0, "collected_landmarks must have at least 1 right hand landmark"

        logging.info("Building TF Record datasets...")
        self.__build_train_dataset(collected_landmarks=collected_landmarks, entries_per_rec=entries_per_rec)
        logging.info("Finished building datasets")

    def __build_train_dataset(self, collected_landmarks, entries_per_rec=300):
        total_classes = np.unique(np.array(self.classes)) # Get all unique classes

        # Start processing and saving images
        maxDatasetSize = len(self.frame_paths)

        os.makedirs(self.output_path, exist_ok=True) # Create output directory if it doesn't exist

        count = 0
        export_count = 0

        content=[]
        
        print(self.frame_paths)
        for i,path in enumerate(self.frame_paths):
            # Get the label for the current image
            label = self.classes[i]

            if os.path.exists(path) == False: # Check if image path is valid/exists
                print("image path does not exist")
                continue
            
            image = cv.imread(path)
            pose_landmarks, hand_landmarks, handedness, _ = self.landmark_processor.get_landmarks(image) # Get landmarks from current image

            print(f"Hand landmarks: {hand_landmarks}")

            if hand_landmarks == [] or len(hand_landmarks) < 1:
                continue # Skip this image if there are no hand landmarks

            print("Hand landmark found")
            
            landmarks = []
            landmarks.extend(hand_landmarks[0][i].x for i in collected_landmarks[1])
            landmarks.extend(pose_landmarks[i].y for i in collected_landmarks[0])

            landmarks.extend(hand_landmarks[0][i].y for i in collected_landmarks[1])
            landmarks.extend(pose_landmarks[i].y for i in collected_landmarks[0])

            landmarks.extend(hand_landmarks[0][i].z for i in collected_landmarks[1])
            landmarks.extend(pose_landmarks[i].z for i in collected_landmarks[0])
            example = self.__serialize(landmarks, label)
            content.append(example)
            count += 1
            if count >= entries_per_rec or i == (maxDatasetSize - 1):
                export_count += 1
                with tf.io.TFRecordWriter(os.path.join(self.output_path, f"train_{export_count:04}_{str(uuid.uuid4())}.tfrecord")) as writer:
                    self.__write(writer, content)
                content = []
                count = 0
            


    def __float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    def __int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def __bytes_feature(self, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def __serialize(self, landmarks, label):
        feature = {
            "landmarks": self.__float_feature(landmarks),
            "target_label": self.__int64_feature(ALPHABET.index(label))
        }

        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
    
    def __write(self, writer, examples):
        for rec in examples:
            writer.write(rec)