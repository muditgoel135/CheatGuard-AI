"""
landmarker.py — MediaPipe model configuration and landmark drawing helpers.

Exposes three pre-configured MediaPipe task objects ready for use in IMAGE mode:
    face_detector_options   — short-range face detection (presence check).
    face_landmark_options   — 478-point face mesh (used for gaze/pose analysis).
    hand_landmark_options   — 21-point hand skeleton for up-to-2 hands.

Drawing helpers (draw_face_landmarks_on_image, draw_hand_landmarks_on_image) modify the passed RGB ndarray in-place and also return it for convenience.

All model paths are resolved relative to BASE_DIR (this file's directory) so the package works regardless of the working directory at launch.
"""

# Import necessary libraries
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult


# Mediapipe setup
BaseOptions = mp.tasks.BaseOptions


# Paths to the models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_to_face_detection_model = os.path.join(
    BASE_DIR, "detection_models", "face_detection_short_range.tflite"
)

path_to_landmark_model = os.path.join(
    BASE_DIR, "detection_models", "face_landmarker.task"
)

path_to_hand_landmark_model = os.path.join(
    BASE_DIR, "detection_models", "hand_landmarker.task"
)


# Define Face Detector and Face Landmarker
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


# Define Hand Landmarker
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


# Mediapipe Detection and landmarking options with model paths
face_detector_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=path_to_face_detection_model),
    running_mode=VisionRunningMode.IMAGE,
)

face_landmark_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=path_to_landmark_model),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

hand_landmark_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=path_to_hand_landmark_model),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
)


# Drawing style objects are built once at import time.  Each call to
# get_default_*_style() allocates new Python objects; caching them here avoids
# thousands of small allocations per second at 30 fps across multiple cameras.
_TESSELATION_STYLE = drawing_styles.get_default_face_mesh_tesselation_style()
_CONTOURS_STYLE = drawing_styles.get_default_face_mesh_contours_style()
_IRIS_STYLE = drawing_styles.get_default_face_mesh_iris_connections_style()
_HAND_LANDMARKS_STYLE = drawing_styles.get_default_hand_landmarks_style()
_HAND_CONNECTIONS_STYLE = drawing_styles.get_default_hand_connections_style()


def draw_hand_landmarks_on_image(
    rgb_image: np.ndarray, detection_result: HandLandmarkerResult
) -> np.ndarray:
    """
    Draws the hand landmarks from the given detection result on the given image.

    :param rgb_image: The image on which to draw the hand landmarks.
    :param detection_result: The result of hand landmark detection, to tell the locations of the hand landmarks.

    :return: The annotated image with hand landmarks drawn on it.
    :rtype: np.ndarray
    """

    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = rgb_image  # draw in-place; caller owns this array

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the hand landmarks on the image.
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=hand_landmarks,
            connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=_HAND_LANDMARKS_STYLE,
            connection_drawing_spec=_HAND_CONNECTIONS_STYLE,
        )

    # Return the annotated image with hand landmarks drawn on it.
    return annotated_image


def draw_face_landmarks_on_image(
    rgb_image: np.ndarray, detection_result: FaceLandmarkerResult
) -> np.ndarray:
    """
    Draws the face landmarks from the given detection result on the given image.

    :param rgb_image: The image on which to draw the face landmarks.
    :param detection_result: The result of face landmark detection, to tell the locations of the face landmarks.

    :return: The annotated image with face landmarks drawn on it.
    :rtype: np.ndarray
    """

    face_landmarks_list = detection_result.face_landmarks
    annotated_image = rgb_image  # draw in-place; caller owns this array

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Three-pass drawing pipeline — each pass uses a different connection set
        # so MediaPipe can apply distinct colours/thicknesses to each layer:
        #   1. Tesselation — the fine triangle mesh covering the whole face surface.
        #   2. Contours    — the bold outline edges (jawline, eyes, lips, etc.).
        #   3. Irises      — small circles around each iris centre point.
        # landmark_drawing_spec=None suppresses individual landmark dots so only
        # the connection lines are rendered, keeping the overlay uncluttered.

        # Pass 1: tesselation
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=_TESSELATION_STYLE,
        )

        # Pass 2: contours
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=_CONTOURS_STYLE,
        )

        # Pass 3: irises (left and right drawn separately as they share the same style)
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=_IRIS_STYLE,
        )

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=_IRIS_STYLE,
        )

    # Return the annotated image with face landmarks drawn on it.
    return annotated_image
