import cv2
import os
import numpy as np

# Tweak these based on how small / large the cards are displayed
AREA_FLOOR = 350
AREA_CELING = 1500

# for blurry / unfocused camera
def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3.5, sigmaY=3.5)
    sharpened = cv2.addWeighted(image, 1.2, blurred, -0.5, 0)
    return sharpened

# preprocesses image for processing
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # converts the gray and blurred image into a binary image (black and white only)
    # uses OTSU method, which automatically tunes the threshold
    _, thresh = cv2.threshold(blur_image, 80, 255, cv2.THRESH_BINARY_INV)
    return thresh

def create_box_contour(contour):
    x, y, w, h = cv2.boundingRect(contour)

    # remove any super skinny boxes
    aspect_ratio = w / float(h)
    if not 0.2 < aspect_ratio < 2.0:
        return None

    box_points = np.array([
        [[x, y]],
        [[x + w, y]],
        [[x + w, y + h]],
        [[x, y + h]]
    ], dtype=np.int32)

    return box_points

# takes in an big image, and finds the contours
# returns the rank and suit contours [box contour, actual shape contour]
def find_cards(thresh_image, original_frame):
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []

    for cnt in contours:
        box_contour = create_box_contour(cnt)
        if box_contour is None:
            continue

        area = cv2.contourArea(box_contour)
        if AREA_FLOOR < area < AREA_CELING:
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(box_contour)
            filtered_contours.append((box_contour, cnt, x, y))  # Store x and y for sorting

    # Sort contours based on proximity to the top-left corner (smallest x + y)
    filtered_contours = sorted(filtered_contours, key=lambda item: (item[3], item[2]))

    # Select the top-left most two contours
    if len(filtered_contours) >= 2:
        return (filtered_contours[0][0], filtered_contours[0][1]), (filtered_contours[1][0], filtered_contours[1][1])
    else:
        # Draw remaining contours for debugging if less than 2 are found
        print(f"Contours found: {len(filtered_contours)}")
        for box_contour, _, _, _ in filtered_contours:
            cv2.drawContours(original_frame, [box_contour], -1, (0, 0, 255), 2)
        return None, None


# loads the templates in ./templates folder into memory
def load_templates(template_dir):
    templates = {'Ranks': {}, 'Suits': {}}

    # Define subdirectories for Ranks and Suits
    rank_dir = os.path.join(template_dir, 'Ranks')
    suit_dir = os.path.join(template_dir, 'Suits')

    # Load Rank Templates
    if os.path.exists(rank_dir):
        for filename in os.listdir(rank_dir):
            if filename.endswith('.jpg'):
                template_name = os.path.splitext(filename)[0]  # e.g., 'A', '2', '3'
                template_path = os.path.join(rank_dir, filename)
                template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
                templates['Ranks'][template_name] = template_img

    # Load Suit Templates
    if os.path.exists(suit_dir):
        for filename in os.listdir(suit_dir):
            if filename.endswith('.jpg'):
                template_name = os.path.splitext(filename)[0]  # e.g., 'hearts', 'spades'
                template_path = os.path.join(suit_dir, filename)
                template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
                templates['Suits'][template_name] = template_img

    return templates

# extracts the image of a contour (box)
def extract_image(contour, image):
    # Extract bounding boxes for rank and suit
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the rank and suit from the original image
    return image[y:y + h, x:x + w]


# finds the best match for imput image vs template
# returns the text of the best match
def find_best_match(input_img, templates, isSuit):
    """
    Finds the best match for the input rank or suit image by using pixel differencing.

    Args:
        input_img (ndarray): The query rank or suit image.
        templates (dict): Dictionary with 'Ranks' and 'Suits' holding preprocessed template images.
        isSuit (bool): If True, matches against suit templates; if False, matches against rank templates.

    Returns:
        str: The name of the best-matching template.
    """
    best_match = "Unknown"
    best_match_diff = 10000  # High initial value
    # print("--START--")
    # Select appropriate template set
    template_set = templates.get('Suits', {}) if isSuit else templates.get('Ranks', {})

    if input_img is not None and len(input_img) != 0:
        # Resize input image to match template size
        for template_name, template_img in template_set.items():
            # Ensure both images are the same size for comparison
            resized_input = cv2.resize(input_img, (template_img.shape[1], template_img.shape[0]))

            # Compute absolute difference
            diff_img = cv2.absdiff(resized_input, template_img)
            diff_score = int(np.sum(diff_img) / 255)

            # Debug: Print the difference score
            # print(f"Template: {template_name}, Difference Score: {diff_score}")

            if diff_score < best_match_diff:
                best_match_diff = diff_score
                best_match = template_name

    # Apply a threshold to avoid false positives
    DIFF_MAX = 2250  # Adjust this threshold based on your testing
    # print(best_match_diff, best_match)
    if best_match_diff > DIFF_MAX:
        return "Unknown"
    # print("--END--")
    return best_match

# resizes image to match either suit or rank dimensions (suit is 70x100, rank is 70x125)
def resize_card_image(image, isSuit):
    """
    Resizes the rank or suit image to standard dimensions.

    Args:
        image (ndarray): The input rank or suit image.
        isSuit (bool): If True, resizes for suit; if False, resizes for rank.

    Returns:
        ndarray: The resized image.
    """
    # Define target dimensions
    target_size = (70, 100) if isSuit else (70, 125)  # (width, height)

    # Resize the image
    resized_image = cv2.resize(image, target_size)

    # if isSuit:
    #     cv2.imwrite('club.jpg', preprocess_image(resized_image))
    #     print("Saved resized suit image as spade.jpg")

    return resized_image
