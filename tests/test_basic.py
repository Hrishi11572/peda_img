import peda_img
import numpy as np

def test_gray_conversion():
    img = peda_img.handle_image("test_images/oppieBW.png")
    thrs_img = peda_img.otsu_thresholding(img, inverse=True)
    peda_img.show_image(thrs_img)

test_gray_conversion()