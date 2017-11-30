import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm


def create_base():
    base = []
    for folder in tqdm(glob('.\pictures\Img\*'), desc='Reading images'):
            if os.path.isdir(folder):
                images = []
                for img_path in glob(folder + '\img*.png'):
                    # print(img_path)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(660).astype(np.float32, copy=False)
                    images.append(img)
                base.append(images)
    return base


def read_sample(path):
    img = cv2.imread(path, 0)
    height, width = img.shape[:2]
    max_height = 900
    max_width = 1600

    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # return img
    return pre_process(img)


def pre_process(image):
    ret, tresholded = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)  # threshold image
    im = cv2.medianBlur(tresholded, 5)  # median blur to eliminate noise
    # cv2.imshow('filtered', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    im2,contours,hierarchy = cv2.findContours(im, 1, 2)  # finding contours

    areas = [cv2.contourArea(c) for c in contours]  # from found contours we choose the biggest one
    max_index = np.argmax(areas)
    letters = []
    x, y, w, h_max = cv2.boundingRect(contours[max_index])  # find bounding box
    h_max /= 2
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # find bounding box
        if h >= h_max:
            letters.append(cv2.bitwise_not(cv2.resize(im[y:y + h, x:x + w], (50, 50), interpolation=cv2.INTER_AREA)))
    return letters


def test_processing(path):
    result = read_sample(path)
    for image in result:
        # cv2.imwrite('pictures\ptest.png', result)
        cv2.imshow('frame', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_base():
    baseN = create_base()
    print (len(baseN))
    print (len(baseN[2]))
    cv2.imwrite('pictures\pbtest.png', baseN[0][2])


test_processing('pictures\phone2.jpg')
