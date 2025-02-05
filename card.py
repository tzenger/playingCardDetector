import cv2

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # converts the gray and blurred image into a binary image (black and white only)
    # uses OTSU method, which automatically tunes the threshold
    _, thresh = cv2.threshold(blur_image, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh