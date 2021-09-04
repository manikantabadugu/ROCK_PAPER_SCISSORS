import mediapipe as mp
import cv2

import sys


import processing_df_image as pdi
import time
import numpy as np
import os
import sys

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)


def landmark_xy(hn_landmarks, image):
    height, width, _ = image.shape
    hand_points = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
                   'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
                   'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

    coord_dict = {}
    for point in hand_points:
        try:
            x = int(hn_landmarks.landmark[getattr(mp_hands.HandLandmark, point)].x * width)
            y = int(hn_landmarks.landmark[getattr(mp_hands.HandLandmark, point)].y * height)
        # z = hn_landmarks.landmark[getattr(mp_hands.HandLandmark, point)].z * height
        except:
            x = np.nan
            y = np.nan
        coord_dict[point + 'x'] = x
        coord_dict[point + 'y'] = y

    return coord_dict


def bbox_landmarks(hn_landmark, image):
    padding = 20
    crop_copy = image.copy()

    x = [landmark.x for landmark in hn_landmark.landmark]
    y = [landmark.y for landmark in hn_landmark.landmark]

    coords = [min(x) * image.shape[1], max(x) * image.shape[1], min(y) * image.shape[0], max(y) * image.shape[0]]
    center = np.array([np.mean(x) * image.shape[1], np.mean(y) * image.shape[0]]).astype('int32')

    dist = [center[0] - coords[0], coords[1] - center[0], center[1] - coords[2], coords[3] - center[1]]
    bb_dim = int(max(dist) + padding)

    start_r = center[1] - bb_dim
    start_c = center[0] - bb_dim
    end_r = center[1] + bb_dim
    end_c = center[0] + bb_dim
    
    if start_r != 0 or start_c != 0:
        crop = crop_copy[start_r:end_r, start_c:end_c]
    elif start_r < 0 and start_c < 0:
        crop = crop_copy[:end_r, :end_c]

    cv2.circle(image, tuple(center), 10, (255, 0, 0), 2)  # for checking the center
    cv2.rectangle(image, (center[0] - bb_dim, center[1] - bb_dim), (center[0] + bb_dim, center[1] + bb_dim),
                  (255, 0, 0), 2)

    return crop

def transform_single_image(image, output_path=""):

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
        if type(image) == str:
            output = os.path.join(output_path, image)
            image = cv2.imread(image)
        else:
            output = os.path.join(output_path, "image.png")
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        list_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    list_land = hand_landmarks.landmark
                    # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    black_image = np.zeros(image.shape)
                    mp_drawing.draw_landmarks(black_image, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec)
                    crop = pdi.bbox_landmarks(hand_landmarks, black_image)
                    kernel = np.ones((5, 5), np.uint8)
                    crop = cv2.dilate(crop, kernel, iterations = 10)
                    crop = np.float32(crop)
                    gray = cv2.cvtColor(255*crop, cv2.COLOR_BGR2GRAY)
                    crop = cv2.resize(gray, (32, 32)) 
                    pdi.bbox_landmarks(hand_landmarks, image)
                    loc_dict = pdi.landmark_xy(hand_landmarks, image)
                except:
                    print("The hand is too close to the camera")
    try:
        return crop
    except:
        return crop


if __name__ == "__main__":
    try:
        image_path = sys.argv[1]
    except:
        "Image path not provided or not valid"
    try:
        save = sys.argv[3]
    except:
        save = 0
    try:
        output_path = sys.argv[4]
    except:
        output_path = ""
    crop = transform_single_image(image_path)
    if save:
        output = os.path.join(output_path, "cropped_"+image_path)
        cv2.imwrite(output, crop)

      