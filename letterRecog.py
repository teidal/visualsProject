import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm


# creates database as a numpy file, with 2 lists: list of pictures and list of labels. picture[i] has label[i]
def create_base():
    label = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
        15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q',
        27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c',
        39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o',
        51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
    }
    pictures1 = []
    labels1 = []
    i = 0
    for folder in tqdm(glob('.\pictures\Img\*'), desc='Reading images'):
            if os.path.isdir(folder):
                for img_path in glob(folder + '\img*.png'):
                    # print(img_path)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (56, 56), interpolation=cv2.INTER_AREA)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(660).astype(np.float32, copy=False)
                    pictures1.append(img)
                    labels1.append(label[i])
                i += 1
    pictures = np.array(pictures1)
    labels = np.array(labels1)
    np.savez('base.npz', pictures=pictures, labels=labels)


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


# apply median blur, manual thresholding, find boundig boxes
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


# for testing
if __name__ == "__main__":
    x = create_base()
    print("Done")
    #print(x[825])
