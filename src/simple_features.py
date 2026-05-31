import cv2
import numpy as np

def extract_color_feature(frame, bbox):
    "Extract a simple color feature from the player's upper body because of the jersey"

    x1, y1, x2, y2 = map(int, bbox)

    # Make sure the coordinates are inside the image.
    h_frame, w_frame = frame.shape[:2]

    x1 = max(0, min(x1, w_frame-1))
    x2 = max(0, min(x2, w_frame-1))
    y1 = max(0, min(y1, h_frame-1))
    y2 = max(0, min(y2, h_frame-1))

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return np.zeros(576, dtype=np.float32)
    
    h, w = crop.shape[:2]

    # Taking only upper body because it is more useful then legs.
    upper_body = crop[int(0.15*h):int(0.60*h), :]

    if upper_body.size == 0:
        return np.zeros(576, dtype=np.float32)
    
    #Convert BGR to HSV which is better in color comparison then the normal RGB/BGR
    hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)

    # Create color histogram
    hist = cv2.calHist(
        [hsv],
        [0,1],
        None,
        [24, 24],
        [0, 180, 0, 256],
    )

    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()

    return hist.astype(np.float32)