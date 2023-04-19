import cv2
import numpy as np


def drawBox(boxes, image):

    cv2.rectangle(image, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (255, 0, 0), 1)
    cv2.imshow("img", image)
    cv2.waitKey()



def cvTest(file_name, x, y, w, h):
    # imageToPredict = cv2.imread("img.jpg", 3)
    imageToPredict = cv2.imread("./sample_dataset/" + file_name)
    print(imageToPredict.shape)

    # Note: flipped comparing to your original code!
    y_ = imageToPredict.shape[0]
    x_ = imageToPredict.shape[1]


    targetSize = 224
    x_scale = targetSize / x_
    y_scale = targetSize / y_
    print(x_scale, y_scale)
    img = cv2.resize(imageToPredict, (targetSize, targetSize));
    print(img.shape)


    # original frame as named values
    (origLeft, origTop, origRight, origBottom) = (x, y, x + w, y + h)


    x = int(np.round(origLeft * x_scale))
    y = int(np.round(origTop * y_scale))
    xmax = int(np.round(origRight * x_scale))
    ymax = int(np.round(origBottom * y_scale))
    print([x, y, xmax - x, ymax - y])
    drawBox([x, y, xmax, ymax], img)
    #drawBox([x , y , (x + w), (y + h)], imageToPredict)

def resize224(image_width, image_height, x, y , w, h):

    y_ = image_height
    x_ = image_width
    targetSize = 224
    x_scale = targetSize / x_
    y_scale = targetSize / y_
    (origLeft, origTop, origRight, origBottom) = (x, y, x + w, y + h)
    x = int(np.round(origLeft * x_scale))
    y = int(np.round(origTop * y_scale))
    xmax = int(np.round(origRight * x_scale))
    ymax = int(np.round(origBottom * y_scale))

    return [x, y, xmax-x, ymax-y]

print(resize224( 427,640,4.3,163.75,327.18,467.81))
cvTest("581258.jpg",4.3,163.75,327.18,467.81)
