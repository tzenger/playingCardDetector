import cv2
from card import *

# Initialize main camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution set to: {int(width)}x{int(height)}")

# LOAD TEMPLATES IN MEMORY
templates = load_templates('./templates')

if not cam.isOpened():
    print("Camera can't open")
    exit()

# Flags for opening display windows
rank_window_flag = False
suit_window_flag = False

while True:
    ret, frame = cam.read()
    better_frame = sharpen_image(frame)
    better_frame = cv2.rotate(better_frame, cv2.ROTATE_180)
    if not ret:
        print("Can't receive frame, exiting...")
        break

    # ----- PREPROCESSING -----
    processed_frame = preprocess_image(better_frame)
    rank_cnt, suit_cnt = find_cards(processed_frame, better_frame)

    # ----- DISPLAY WINDOWS -----
    cv2.imshow('Camera Feed', better_frame)
    cv2.imshow('Preprocess', processed_frame)

        # ----- RANK / SUIT DETECTION -----
    if rank_cnt is not None and suit_cnt is not None:
        rank_im = extract_image(rank_cnt[0], better_frame)
        suit_im = extract_image(suit_cnt[0], better_frame)

        rank = find_best_match(preprocess_image(resize_card_image(rank_im, False)), templates, False)
        suit = find_best_match(preprocess_image(resize_card_image(suit_im, True)), templates, True)

        # Display the rank and suit on the better_frame
        display_text = f"Detected: {rank} of {suit}"
        cv2.putText(better_frame, display_text, 
                (10, better_frame.shape[0] - 10),  # Bottom-left corner of the frame
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                (0, 255, 0), 2, cv2.LINE_AA)

        # Show rank and image windows
        cv2.imshow('Rank Image', preprocess_image(rank_im))
        cv2.imshow('Suit Image', preprocess_image(suit_im))
        rank_window_flag = True
        suit_window_flag = True    

    elif rank_window_flag:
        # Destroy rank and image windows
        cv2.destroyWindow('Rank Image')
        cv2.destroyWindow('Suit Image')
        rank_window_flag = False
        suit_window_flag = False

    cv2.imshow('Camera Feed', better_frame)

    # KEEP THIS IN IDK WHY IF I TAKE IT OUT IT BREAKS
    key = cv2.waitKey(1) & 0xFF  # Wait for 1ms and mask to get last 8 bitsqqq


cam.release()

cv2.destroyAllWindows()