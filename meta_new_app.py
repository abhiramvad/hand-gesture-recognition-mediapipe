#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F

from utils import CvFpsCalc  # Ensure this is in your utils module
from transformer_MAML import MAMLTransformer, TransformerClassifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def main():
    args = get_args()
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = TransformerClassifier(input_dim=2, hidden_dim=64, num_classes=11).to(device)
    maml_model = MAMLTransformer(base_model, inner_lr=0.01, inner_steps=1).to(device)
    maml_model.base_model.load_state_dict(torch.load('model/keypoint_classifier/maml_transformer.pt', map_location=device))
    maml_model.eval()

    # Load labels
    # Load labels and handle missing class labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    # Add placeholder labels if new class exists
    num_model_classes = 11  # Update to match model
    if len(keypoint_classifier_labels) < num_model_classes:
        keypoint_classifier_labels += [f'Class {i}' for i in range(len(keypoint_classifier_labels), num_model_classes)]


    cvFpsCalc = CvFpsCalc(buffer_len=10)
    point_history = deque(maxlen=16)

    mode = 0
    number = -1

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                if len(landmark_list) != 21:
                    continue

                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                keypoints = np.array(pre_processed_landmark_list, dtype=np.float32).reshape(21, 2)

                if mode == 1 and number != -1:
                    row = [number] + pre_processed_landmark_list
                    with open('keypoint.csv', 'a', newline="") as f:
                        csv.writer(f).writerow(row)

                # Meta-learning prediction
                keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)
                support_x = torch.stack([keypoints_tensor.squeeze()] * 5)
                support_y = torch.tensor([10] * 5).to(device)
                query_x = keypoints_tensor
                query_y = torch.tensor([10]).to(device)

                tasks = [(support_x, support_y, query_x, query_y)]
                _, _ = maml_model(tasks)

                output = maml_model.base_model(query_x)
                hand_sign_id = output.argmax(dim=1).item()

                point_history.append(landmark_list[8])

                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness,
                                             keypoint_classifier_labels[hand_sign_id], "")
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

# Utility functions (assumed unchanged)
from new_app import calc_bounding_rect, calc_landmark_list, pre_process_landmark, draw_bounding_rect, draw_landmarks, draw_info_text, draw_point_history, draw_info

if __name__ == '__main__':
    main()