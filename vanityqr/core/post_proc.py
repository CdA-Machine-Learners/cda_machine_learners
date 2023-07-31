import numpy as np
from PIL import Image, ImageDraw

import cv2, ctypes

def draw_corners( image: Image, box_size=16, clone=False ):
    w, h = image.width, image.height

    # iterate through QR codes to draw into target
    draw = ImageDraw.Draw(image)

    # Draw the corners
    draw.rectangle( draw_ul( w, h, box_size, 0, 0, 8, 8 ), fill="white")
    draw.rectangle( draw_ul( w, h, box_size, 0, 0, 7, 7 ), fill="black")
    draw.rectangle( draw_ul( w, h, box_size, 1, 1, 5, 5 ), fill="white")
    draw.rectangle( draw_ul( w, h, box_size, 2, 2, 3, 3 ), fill="black")

    draw.rectangle( draw_ur( w, h, box_size, 0, 0, 8, 8 ), fill="white")
    draw.rectangle( draw_ur( w, h, box_size, 0, 0, 7, 7 ), fill="black")
    draw.rectangle( draw_ur( w, h, box_size, 1, 1, 5, 5 ), fill="white")
    draw.rectangle( draw_ur( w, h, box_size, 2, 2, 3, 3 ), fill="black")

    draw.rectangle( draw_ll( w, h, box_size, 0, 0, 8, 8 ), fill="white")
    draw.rectangle( draw_ll( w, h, box_size, 0, 0, 7, 7 ), fill="black")
    draw.rectangle( draw_ll( w, h, box_size, 1, 1, 5, 5 ), fill="white")
    draw.rectangle( draw_ll( w, h, box_size, 2, 2, 3, 3 ), fill="black")


def draw_ul( width, height, bs, x, y, w, h):
    return [(x * bs, y * bs), ((x + w) * bs, (y + h) * bs)]

def draw_ur( width, height, bs, x, y, w, h):
    x1, y1 = (width - x * bs, y * bs)
    x2, y2 = (width - (x + w) * bs, (y + h) * bs)
    return [(x2, y1), (x1, y2)]

def draw_ll( width, height, bs, x, y, w, h):
    x1, y1 = (x * bs, height - y * bs)
    x2, y2 = (x + w) * bs, height - (y + h) * bs
    return [(x1, y2), (x2, y1)]


def value_shift( image: Image, qr_code: Image, box_size=16, border=0 ):
    # Load the shared library
    lib = ctypes.CDLL('core/worker.so')

    # Declare the argument and return types of the C function
    lib.c_function.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.py_object,
        ctypes.py_object,
    ]
    lib.c_function.restype = None  # No return value

    # Create a NumPy array
    np_img = np.array(image.convert("RGB"), dtype=np.uint8)
    np_qr = np.array(qr_code.convert("L"), dtype=np.uint8)

    # Call the C function
    lib.c_function(0.28, 0.1, 0.08, np_img, np_qr)
    #lib.c_function(0.15, 0.1, 0.05, np_img, np_qr)

    # Convert back to a PIL image
    return Image.fromarray(np_img)