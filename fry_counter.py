import cv2 as cv
import numpy as np

# Function for opening image
def get_img(img_path, grayscale=True):
    img = cv.imread(img_path)
    if grayscale:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

# Function for cropping image
def crop_img(img):
    img_centerX = img.shape[1] // 2
    img_centerY = img.shape[0] // 2

    # Initialize mask to use
    mask = np.zeros(img.shape, dtype=np.uint8)

    # Crop the image
    region_to_cut = cv.ellipse(mask, (img_centerX, img_centerY), (547, 547), 0, 0, 360, 255, -1)
    cropped_img = cv.bitwise_or(img, img, mask=region_to_cut)
    return cropped_img[:, img_centerX-600:img_centerX+600]

# Gaussian Blur Function
def gaussian_blur(img, kernel_a, kernel_b):
    return cv.GaussianBlur(img, (kernel_a, kernel_b), 0)

# Remove the background from the fry image
def img_diff(img, bg_img):
    return cv.absdiff(img, bg_img)

# Image Segmentation Using Thresholding Function
def threshold_img(diff_img, type='OTSU', block=0, c=0):
    if type == 'OTSU':
        _, thresh_img = cv.threshold(diff_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return thresh_img
    return cv.adaptiveThreshold(diff_img, 256, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block, c)

# Histogram Counter
def histogram_counter(thres_img, pixel_to_count='white'):
    if pixel_to_count == 'white':
        return np.count_nonzero(thres_img == 255)
    return np.count_nonzero(thres_img == 0)

# Countour Detection
def contour_detection(thres_img):
    contours, _ = cv.findContours(thres_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

def fry_counter(img_path):
    img = get_img(img_path)
    bg_img = get_img('./IMAGES/100/background.jpg')

    # Crop the image
    img = crop_img(img)
    bg_img = crop_img(bg_img)

    # Image Difference
    diff_img = img_diff(img, bg_img)

    # Adaptive Gaussian Threshold
    thres_img = threshold_img(diff_img, type="ADAPTIVE", block=37, c=5)

    # Histogram Counter
    white_pixels = histogram_counter(thres_img)

    # Length of Contours
    len_contours = len(contour_detection(thres_img))

    cv.imshow('img_diff', diff_img)

    """
        FRY COUNT
        100, if white_pixels is between 1265864 and 1270473 and contour length is between 748 to 893
        200, if white_pixels is between 1221275 and 1229159 and contour length is between 2233 to 2571
        300, if white_pixels is between 1196588 and 1216145 and contour length is between 2516 to 2955
        400, if white_pixels is between 1160103 and 1172078 and contour length is between 2982 to 3476
    """

    if white_pixels >= 1265864 and white_pixels <= 1270473 and len_contours >= 748 and len_contours <= 893:
        return 100
    elif white_pixels >= 1221275 and white_pixels <= 1229159 and len_contours >= 2233 and len_contours <= 2571:
        return 200
    elif white_pixels >= 1196588 and white_pixels <= 1216145 and len_contours >= 2516 and len_contours <= 2955:
        return 300
    elif white_pixels >= 1160103 and white_pixels <= 1172078 and len_contours >= 2982 and len_contours <= 3476:
        return 400
    else:
        raise Exception("Fry Count Cannot Be Determined")


def fry_detector(img_path, fry_count):
    img_kernels = {
        100: (9, 9),
        200: (13, 5),
        300: (5, 5),
        400: (11, 11)
    }
    bg_kernels = {
        100: (5, 13),
        200: (7, 3),
        300: (5, 11),
        400: (9, 3)
    }

    # Get the images
    img = get_img(img_path)
    bg_img = get_img(f'./IMAGES/{fry_count}/background.jpg')

    # Crop the image
    img = crop_img(img)
    bg_img = crop_img(bg_img)

    # Gaussian Blur
    img_kernel = img_kernels.get(fry_count, img_kernels[100])
    bg_kernel = bg_kernels.get(fry_count, bg_kernels[100])
    img = gaussian_blur(img, img_kernel[0], img_kernel[1])
    bg_img = gaussian_blur(bg_img, bg_kernel[0], bg_kernel[1])

    # Image Difference
    diff_img = img_diff(img, bg_img)

    # OTSU Threshold
    thres_img = threshold_img(diff_img, type="OTSU")

    # Draw Contours
    contours = contour_detection(thres_img)

    # Display the image
    img_final = cv.drawContours(img, contours, -1, (0, 255, 0), 1)
    cv.namedWindow('Fry Counter and Detector', cv.WINDOW_NORMAL)
    cv.resizeWindow('Fry Counter and Detector', 600, 540)
    cv.putText(img_final, f'~Fry Count: {fry_count}', (600//2,540//2), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow('Fry Counter and Detector', img_final)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    while True:
        img_path = input("Drop photo here: ")
        fry_count = fry_counter(img_path[1:-1])
        fry_detector(img_path[1:-1], fry_count)
