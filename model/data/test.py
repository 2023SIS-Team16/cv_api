from dataset_builder import DatasetBuilder
from inference_testing.landmarks import LandmarkProcessor

A_SIZE = 278
B_SIZE = 401
C_SIZE = 364
D_SIZE = 355
E_SIZE = 337
F_SIZE = 413
G_SIZE = 448
H_SIZE = 546
I_SIZE = 546
K_SIZE = 487
L_SIZE = 448
M_SIZE = 423
N_SIZE = 562
O_SIZE = 548
P_SIZE = 763
Q_SIZE = 639
R_SIZE = 491
S_SIZE = 613
U_SIZE = 635
W_SIZE = 631
Y_SIZE = 682

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'w', 'y']

image_paths = []
classes = []

import os
base_path = "/Users/jon/development/university/sis/videos/alphabet"
for letter in letters:
    letter_path = os.path.join(base_path, letter)
    for file in os.listdir(letter_path):
        if not file.endswith('.png'): continue
        image_paths.append(os.path.join(letter_path, file))
        classes.append(letter)

processor = LandmarkProcessor(
    pose_landmarker="/Users/jon/development/university/sis/models/pose_landmarker_full.task",
    hand_landmarker="/Users/jon/development/university/sis/models/hand_landmarker.task",
    face_landmarker="/Users/jon/development/university/sis/models/face_landmarker.task"
)

builder = DatasetBuilder(frame_paths=image_paths, output_path="/Users/jon/development/university/sis/datasets/output2/", classes=classes, landmark_processor=processor)

builder.build_train(collected_landmarks=[
    # [12, 14, 16, 18, 20, 22, 11, 13, 15, 17, 19, 21],
    [],
    list(range(0, 21)), 
    []], entries_per_rec=300)

