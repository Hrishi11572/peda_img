import peda_img
import numpy as np

def test_gray_conversion():
    img = peda_img.handle_image("test_images/satellite.png")
    gray = peda_img.convert_to_gray(img)
    peda_img.plot_img_hist(img, channel="all")

test_gray_conversion()