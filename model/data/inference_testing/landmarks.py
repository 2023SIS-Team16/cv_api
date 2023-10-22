import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class LandmarkProcessor:
    def __init__(self, pose_landmarker, hand_landmarker, face_landmarker):
        self.pose_landmarker_path = pose_landmarker
        self.hand_landmarker_path = hand_landmarker
        self.face_landmarker_path = face_landmarker

        self.__init_pose_landmarker()
        self.__init_hand_landmarker()
        self.__init_face_landmarker()

    def __init_face_landmarker(self):
        BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.face_options = FaceLandmarkerOptions(
            base_options = BaseOptions(model_asset_path = self.face_landmarker_path),
            running_mode=VisionRunningMode.IMAGE
        )
    
    def __init_pose_landmarker(self):
        BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker

        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.pose_options = PoseLandmarkerOptions(
            base_options = BaseOptions(model_asset_path = self.pose_landmarker_path),
            running_mode=VisionRunningMode.IMAGE
        )

    def __init_hand_landmarker(self):
        BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker

        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.hand_options = HandLandmarkerOptions(
            base_options = BaseOptions(model_asset_path = self.hand_landmarker_path),
            running_mode = VisionRunningMode.IMAGE,
            num_hands = 1,
        )

    def get_landmarks(self, image_arr):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)

        with self.PoseLandmarker.create_from_options(self.pose_options) as pose_landmarker:
            pose_landmarks = pose_landmarker.detect(mp_image)


        with self.HandLandmarker.create_from_options(self.hand_options) as hand_landmarker:
            hand_landmarks = hand_landmarker.detect(mp_image)

        with self.FaceLandmarker.create_from_options(self.face_options) as face_landmarker:
            face_landmarks = face_landmarker.detect(mp_image)
        
        # [result[0].pose_landmarks[0][landmark] for landmark in pose_desired_landmarks_indices]
        filtered_pose_landmarks = pose_landmarks.pose_landmarks[0] if pose_landmarks.pose_landmarks != None and len(pose_landmarks.pose_landmarks) > 0 else []
        filtered_hand_landmarks = []

        for hand in hand_landmarks.hand_landmarks:
            filtered_hand_landmarks = filtered_hand_landmarks + [hand]
            
        
        return filtered_pose_landmarks, filtered_hand_landmarks, hand_landmarks.handedness, face_landmarks.face_landmarks[0] if len(face_landmarks.face_landmarks) > 0 else []
            
