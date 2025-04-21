#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import torch
from torch.func import functional_call

from utils import CvFpsCalc
from maml_graph_transformer import MAMLGraphTransformer
from graph_transformer import GraphTransformerClassifier
import new_app as app_utils  # helper functions


def get_args():
    parser = argparse.ArgumentParser(description="Live MAML‑Graph Transformer Hand Gesture App")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=960, help="Capture width")
    parser.add_argument("--height", type=int, default=540, help="Capture height")
    parser.add_argument("--use_static_image_mode", action="store_true", help="Static image mode for MediaPipe")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7, help="Min detection confidence")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5, help="Min tracking confidence")
    return parser.parse_args()


def select_mode(key, mode, label):
    if key == ord('n'):
        mode = 0; label = -1
    elif key == ord('k'):
        mode = 1
    elif key == ord('h'):
        mode = 2
    if mode == 1:
        if 48 <= key <= 57:
            label = key - 48
        elif key == ord('a'):
            label = 10
    return label, mode


def main():
    args = get_args()

    # Camera setup
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 11

    # Load base GraphTransformer
    base_model = GraphTransformerClassifier(input_dim=2, hidden_dim=64, num_classes=num_classes)
    base_model.load_state_dict(torch.load(
        'model/keypoint_classifier/maml_graph_transformer.pt', map_location=device
    ))
    base_model.to(device).eval()

    # MAML wrapper with stronger adaptation
    maml_model = MAMLGraphTransformer(base_model, inner_lr=0.2, inner_steps=30)
    maml_model.to(device).eval()

    # Load labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        labels = [row[0] for row in csv.reader(f)]
    if len(labels) < num_classes:
        labels += [f'Class {i}' for i in range(len(labels), num_classes)]

    # Prepare negative pool from existing classes (0–9)
    data = np.loadtxt('model/keypoint_classifier/keypoint.csv', delimiter=',', dtype=np.float32)
    X_all = torch.tensor(data[:,1:], dtype=torch.float32).view(-1,21,2).to(device)
    y_all = torch.tensor(data[:,0], dtype=torch.long).to(device)
    mask = y_all < 10
    X_pool = X_all[mask]
    y_pool = y_all[mask]

    cvFpsCalc     = CvFpsCalc(buffer_len=10)
    point_history = deque(maxlen=16)

    # Few-shot support buffers per class
    k_shot   = 5   # support shots for new class
    k_other  = 3   # negative shots per old class
    support_buffers = {i: deque(maxlen=k_shot) for i in range(num_classes)}

    mode        = 0
    label       = -1
    adapted_params    = None
    adaptation_label  = None
    prev_mode   = 0

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        label, mode = select_mode(key, mode, label)

        # Reset adaptation if mode or label changes
        if mode != 2 or adaptation_label != label:
            adapted_params = None
            adaptation_label = None
        
        # Trigger one-time adaptation on mode entry
        if prev_mode != 2 and mode == 2 and label >= 0 and len(support_buffers[label]) >= k_shot:
            # Build positive supports
            pos_x = torch.cat(list(support_buffers[label]))        # (k_shot,21,2)
            pos_y = torch.tensor([label]*k_shot, device=device)    # (k_shot,)

            # Build multi-shot negatives for each old class
            neg_x_list, neg_y_list = [], []
            for c in range(10):
                idxs = (y_pool == c).nonzero().view(-1)
                choices = idxs[torch.randperm(len(idxs))[:k_other]]
                neg_x_list.append(X_pool[choices])  # (k_other,21,2)
                neg_y_list.append(y_pool[choices])  # (k_other,)
            neg_x = torch.cat(neg_x_list, dim=0)
            neg_y = torch.cat(neg_y_list, dim=0)

            # Combine into an 11-way support set
            support_x = torch.cat([pos_x, neg_x], dim=0)
            support_y = torch.cat([pos_y, neg_y], dim=0)

            # One-time adaptation
            adapted_params = maml_model.adapt(support_x, support_y)
            adaptation_label = label

        prev_mode = mode

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        if results.multi_hand_landmarks:
            for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = app_utils.calc_bounding_rect(debug_image, lm)
                landmark_list = app_utils.calc_landmark_list(debug_image, lm)
                if len(landmark_list) != 21:
                    continue

                pts = np.array(app_utils.pre_process_landmark(landmark_list), dtype=np.float32).reshape(21, 2)

                # Collect supports in mode 1
                if mode == 1 and label >= 0:
                    pt = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).to(device)
                    support_buffers[label].append(pt)
                    # Persist to CSV
                    row = [label] + pts.flatten().tolist()
                    with open('model/keypoint_classifier/keypoint.csv', 'a', newline='') as f:
                        csv.writer(f).writerow(row)

                # Build query
                query_x = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).to(device)

                # Use adapted params if available, else base
                if adapted_params is not None and mode == 2 and adaptation_label == label:
                    q_logits = functional_call(base_model, adapted_params, (query_x,))
                else:
                    q_logits = base_model(query_x)

                pred = q_logits.argmax(dim=1).item()

                # Draw annotations
                point_history.append(landmark_list[8])
                debug_image = app_utils.draw_bounding_rect(True, debug_image, brect)
                debug_image = app_utils.draw_landmarks(debug_image, landmark_list)
                debug_image = app_utils.draw_info_text(
                    debug_image, brect, handedness,
                    labels[pred], f"mode={mode}, idx={label}"
                )
        else:
            point_history.append([0, 0])

        debug_image = app_utils.draw_point_history(debug_image, point_history)
        debug_image = app_utils.draw_info(debug_image, fps, mode, label)
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
