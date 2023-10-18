from dataset_builder import DatasetBuilder
from inference_testing.landmarks import LandmarkProcessor

image_paths = [
    "/Users/jon/development/university/sis/videos/fig_1/frame_0028.png",
    "/Users/jon/development/university/sis/videos/fig_1/frame_0029.png",
    "/Users/jon/development/university/sis/videos/fig_1/frame_0030.png",
    "/Users/jon/development/university/sis/videos/fig_1/frame_0031.png",
]

classes = [
    'f', 'f', 'f', 'f'
]

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

