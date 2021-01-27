import cv2
import numpy as np


# This function can resize to any shape, but was built to resize to 84x84
def process_image(image, shape=(84, 84)):
    """Preprocesses a 210x160x3 frame to 84x84x1 grayscale

    Arguments:
        frame: The frame to process.  Must have values ranging from 0-255

    Returns:
        The processed frame
    """
    image = image.astype(np.uint8)  # cv2 requires np.uint8

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image[34:34+160, :160]  # crop image
    image = cv2.resize(image, shape, interpolation=cv2.INTER_NEAREST)
    image = image.reshape((*shape, 1))

    return image

