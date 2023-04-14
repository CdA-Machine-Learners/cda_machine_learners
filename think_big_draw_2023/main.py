import numpy as np
import cv2, math, random


## Neural Network stuff here

## Completed NN

def image_to_array(img, size=28):
    # Convert image to array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size, size))
    #img = cv2.bitwise_not(img)
    #img = img / 255.0
    #img = img.reshape(1, size, size, 1)
    cv2.imshow("Before NN", img)

    ret = []
    for i in range(size):
        for j in range(size):
            ret.append(math.ceil(img[i][j] / 255))

    return ret


#create a 512x512 black image
nn_img = np.zeros((1024,256,3), np.uint8)
draw = np.zeros((512,512,3), np.uint8)

def drawCircles(img, ary):
    #img = img.copy()

    # Calc some dimensions
    init = 30
    height = img.shape[0]
    step = (height - init * 2) / len(ary)

    #draw a circle
    x = math.floor(img.shape[1] / 2)
    y = step / 2 + init / 2
    for t in ary:
        color = (0, math.floor(t * 255), math.floor((1 - t) * 255))

        #non filled circle
        cv2.circle(img, (x, math.floor(y)), math.floor(step * 0.3), (255,0,0), 3)
        #filled circle
        cv2.circle(img, (x, math.floor(y)), math.floor(step * 0.3), color, -1)

        y += step

    return img

def draw_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_LBUTTONDBLCLK:
        (w,h,d) = draw.shape
        cv2.rectangle(draw, (0,0), (w,h), (0,0,0), -1)

    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(draw, (x, y), 8, (255, 255, 255), -1)

        # Run the neural network
        print( image_to_array(draw) )


        ary = [random.random() for i in range(10)]
        drawCircles( nn_img, ary)

# First draw
ary = [random.random() for i in range(10)]
drawCircles( nn_img, ary)

cv2.namedWindow(winname="Draw a number")
cv2.setMouseCallback("Draw a number", draw_mouse)

while True:
    cv2.imshow("Draw a number", draw)
    cv2.imshow("AI Output", nn_img)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cv2.destroyAllWindows()