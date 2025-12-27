import numpy as np 
from .io import convert_to_gray


def threshold_image(img: np.ndarray | None = None, thresholdValue : int = 0, inverse : bool = False)->np.ndarray:
    '''
    Docstring for threshold_image
    
    :param img: The image as np.ndarray
    :type img: np.ndarray | None
    :param thresholdValue: the pixel value at which you want to threshold the image 
    :type thresholdValue: int
    :param inverse: if you want the blacks to become the whites in a thresholded image
    :type inverse: bool
    :return: a thresholded image as np.ndarray 
    :rtype: ndarray[_AnyShape, dtype[Any]]
    '''
    if img is None: 
        raise ValueError("Please enter an image and a threshold value")
    
    if not (0 <= thresholdValue <= 255):
        raise ValueError("thresholdValue must be between 0 and 255.")
    
    if img.ndim == 2: 
        gray_img = img 
    else:
        # convert to grayscale first 
        try:
            gray_img = convert_to_gray(img=img)
        except Exception as e: 
            raise IOError(f"Can't convert to grayscale : {e}")
    
    if not inverse:
        result = np.where(gray_img < thresholdValue, 0, 255)
    else: 
        result = np.where(gray_img < thresholdValue, 255, 0)
    
    return result.astype(np.uint8)