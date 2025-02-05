import cv2

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # converts the gray and blurred image into a binary image (black and white only)
    # uses OTSU method, which automatically tunes the threshold
    _, thresh = cv2.threshold(blur_image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh



def find_cards(thresh_image):
    # RETR_EXTERNAL is the mode that retrieve's outermost contours
    # CHAIN_APPROX_SIMPLE is the contour approx. method, removing redundant points along contour
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_cards = []

    for cnt in contours:
        # calculates the area within a contour
        area = cv2.contourArea(cnt)

        if area > 5000: # filter out small contours | 5000 can be tweaked
            perimeter = cv2.arcLength(cnt, True)

            # (original contour, approximation accuracy (small is detailed), closed contour?)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            # if approximation has 4 sides (quadrilateral)
            if len(approx) == 4:
                detected_cards.append(approx)

    return detected_cards