from dataset_builder import DatasetBuilder
from inference_testing.landmarks import LandmarkProcessor

A_SIZE = 278
B_SIZE = 401

image_paths = []
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/a/frame_{i:04}.png" for i in range(1, A_SIZE+1)])
image_paths.extend([f"/Users/jon/development/university/sis/videos/alphabet/b/frame_{i:04}.png" for i in range(1, B_SIZE+1)])

classes = []
classes.extend(["a" for _ in range(0, A_SIZE)])
classes.extend(["b" for _ in range(0, B_SIZE)])

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

