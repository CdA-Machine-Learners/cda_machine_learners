import io

from PIL import Image, ImageDraw
import numpy as np

from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer
from qrcode.image.styles.colormasks import RadialGradiantColorMask

import qrcode, cv2


def render(data, embeded=None, box_size=16, border=0, err=qrcode.constants.ERROR_CORRECT_H):
    qr = qrcode.QRCode(
        error_correction=err,
        border=border,
        box_size=box_size,
    )

    # Add the URL
    qr.add_data(data)

    # Store the data
    return qr.make_image(
        image_factory=StyledPilImage,
        embeded_image_path=embeded,
        # back_color=(255, 195, 235),
        # fill_color=(55, 95, 35),
    )


# Just a test function, kinda doesn't work right now
def is_valid( image: Image, url ):
    npi = np.array(image)

    # Initialize the QR code detector
    qr_code_detector = cv2.QRCodeDetector()
    data, bbox, _ = qr_code_detector.detectAndDecode( npi)
    print("SSSS")
    print( data )
    print( bbox )
    print( _ )
    print("FFFF")

    return bbox is not None # data == url and bbox is not None
