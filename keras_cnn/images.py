import cv2


def read_grayscale_image(filename):
    img = cv2.imread(filename, 0)
    if img is None:
        raise ValueError("File not found")
    return img


def read_color_image(filename):
    img = cv2.imread(filename)
    if img is None:
        raise ValueError("File not found")
    return img


def read_as_is_image(filename):
    img = cv2.imread(filename, -1)
    if img is None:
        raise ValueError("File not found")
    return img


def save_image(img, filename):
    cv2.imwrite(filename, img)


def show_image(img, ms = 0):
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.imshow("Window", img)
    cv2.waitKey(ms)
    cv2.destroyAllWindows()
