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

image_paths = []
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/a/frame_{i:04}.png" for i in range(1, A_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/b/frame_{i:04}.png" for i in range(1, B_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/c/frame_{i:04}.png" for i in range(1, C_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/d/frame_{i:04}.png" for i in range(1, D_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/e/frame_{i:04}.png" for i in range(1, E_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/f/frame_{i:04}.png" for i in range(1, F_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/g/frame_{i:04}.png" for i in range(1, G_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/h/frame_{i:04}.png" for i in range(1, H_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/i/frame_{i:04}.png" for i in range(1, I_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/k/frame_{i:04}.png" for i in range(1, K_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/l/frame_{i:04}.png" for i in range(1, L_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/m/frame_{i:04}.png" for i in range(1, M_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/n/frame_{i:04}.png" for i in range(1, N_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/o/frame_{i:04}.png" for i in range(1, O_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/p/frame_{i:04}.png" for i in range(1, P_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/q/frame_{i:04}.png" for i in range(1, Q_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/r/frame_{i:04}.png" for i in range(1, R_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/s/frame_{i:04}.png" for i in range(1, S_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/u/frame_{i:04}.png" for i in range(1, U_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/w/frame_{i:04}.png" for i in range(1, W_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/y/frame_{i:04}.png" for i in range(1, Y_SIZE+1)])



classes = []
classes.extend(["a" for _ in range(0, A_SIZE)])
classes.extend(["b" for _ in range(0, B_SIZE)])
classes.extend(["c" for _ in range(0, C_SIZE)])
classes.extend(["d" for _ in range(0, D_SIZE)])
classes.extend(["e" for _ in range(0, E_SIZE)])
classes.extend(["f" for _ in range(0, F_SIZE)])
classes.extend(["g" for _ in range(0, G_SIZE)])
classes.extend(["h" for _ in range(0, H_SIZE)])
classes.extend(["i" for _ in range(0, I_SIZE)])
classes.extend(["k" for _ in range(0, K_SIZE)])
classes.extend(["l" for _ in range(0, L_SIZE)])
classes.extend(["m" for _ in range(0, M_SIZE)])
classes.extend(["n" for _ in range(0, N_SIZE)])
classes.extend(["o" for _ in range(0, O_SIZE)])
classes.extend(["p" for _ in range(0, P_SIZE)])
classes.extend(["q" for _ in range(0, Q_SIZE)])
classes.extend(["r" for _ in range(0, R_SIZE)])
classes.extend(["s" for _ in range(0, S_SIZE)])
classes.extend(["u" for _ in range(0, U_SIZE)])
classes.extend(["w" for _ in range(0, W_SIZE)])
classes.extend(["u" for _ in range(0, Y_SIZE)])

processor = LandmarkProcessor(
    pose_landmarker="/Users/jon/development/university/sis/models/pose_landmarker_full.task",
    hand_landmarker="/Users/jon/development/university/sis/models/hand_landmarker.task",
    face_landmarker="/Users/jon/development/university/sis/models/face_landmarker.task"
)

builder = DatasetBuilder(frame_paths=image_paths, output_path="/Users/jon/development/university/sis/datasets/output", classes=classes, landmark_processor=processor)

builder.build_train(collected_landmarks=[
    [12, 14, 16, 18, 20, 22, 11, 13, 15, 17, 19, 21],
    list(range(0, 21)), 
    []], entries_per_rec=300)

