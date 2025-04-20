import argparse
import copy
import cv2 as cv
import numpy as np
import torch
from torch_geometric.data import Data
import mediapipe as mp

from app import (
    draw_landmarks,
    draw_bounding_rect,
    draw_info_text,
    draw_point_history,
    draw_info,
    calc_bounding_rect,
    calc_landmark_list,
    pre_process_landmark,
    select_mode
)
from gcn import GCNClassifier, get_edge_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    args = parser.parse_args()

    # Load GCN model
    NUM_CLASSES = 5  # adjust if needed
    model = GCNClassifier(input_dim=2, hidden_dim=64, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load('model/keypoint_classifier/gcn.pt', map_location='cpu'))
    model.eval()

    # Prepare graph edges
    edge_index = get_edge_index()

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Camera
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    point_history = []
    mode = 0
    number = -1
    fps_timer = cv.getTickCount()

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.copy(image)
        results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                brect = calc_bounding_rect(image, hand_landmarks)
                landmark_list = calc_landmark_list(image, hand_landmarks)
                pre_proc = pre_process_landmark(landmark_list)

                # Build graph data
                x = torch.tensor(pre_proc, dtype=torch.float).view(21, 2)
                data = Data(x=x, edge_index=edge_index)

                with torch.no_grad():
                    out = model(data.x, data.edge_index, batch=torch.zeros(21, dtype=torch.long))
                    hand_sign_id = int(out.argmax(dim=1))

                # Draw results
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    f"GCN:{hand_sign_id}",
                    ""
                )

        # Display point history and FPS
        debug_image = draw_point_history(debug_image, point_history)
        fps = cv.getTickFrequency() / (cv.getTickCount() - fps_timer)
        fps_timer = cv.getTickCount()
        debug_image = draw_info(debug_image, fps, mode, number)

        cv.imshow('Hand Gesture Recognition (GCN)', debug_image)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
