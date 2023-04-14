import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def crop_img(img):
    img_centerX = img.shape[1] // 2
    img_centerY = img.shape[0] // 2
    mask = np.zeros(img.shape, dtype=np.uint8)
    region_to_cut = cv.ellipse(mask, (img_centerX, img_centerY), (547, 547), 0, 0, 360, 255, -1)
    cropped_img = cv.bitwise_or(img, img, mask=region_to_cut)
    return cropped_img[:, img_centerX-600:img_centerX+600]

# For colored image only
def crop_img2(img):
    img_centerX = img.shape[1] // 2
    return img[:, img_centerX-600:img_centerX+600]

def invert_img(img):
    return cv.bitwise_not(img)

def count_black_pixels(thresh):
    return np.count_nonzero(thresh == 0)

def init_bg(c=100):
    bg_img = cv.imread(f'./IMAGES/{c}/background.jpg')
    bg_img = cv.cvtColor(bg_img, cv.COLOR_BGR2GRAY)
    bg_img_cropped = crop_img(bg_img)
    return invert_img(bg_img_cropped)

def counter():
    bg_img = init_bg()

    # User drops a photo
    img_path = input("Drop photo here: ")
    img = cv.imread(img_path[1:-1])

    # Process Image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_cropped = crop_img(img_gray)
    img_inv = invert_img(img_cropped)

    # Difference and Threshold
    diff_img = cv.absdiff(img_inv, bg_img)
    # remove white specks using Adaptive Gaussian Thresholding
    thresh = cv.adaptiveThreshold(invert_img(diff_img), 256, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31,5)

    # Black pixel count
    black_pixel_count = count_black_pixels(thresh)
    # Contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    """
        FRY COUNT
        100, if black_pixel_count is between 19963 and 21307 and contour length is between 176 to 228
        200, if black_pixel_count is between 62611 and 67718 and contour length is between 1243 to 1436
        300, if black_pixel_count is between 77774 and 87826 and contour length is between 1238 to 1580
        400, if black_pixel_count is between 102196 and 110531 and contour length is between 1409 to 1590
    """
    if 19963 <= black_pixel_count <= 21307 and 176 <= len(contours) <= 228:
        return 100, img
    elif 62611 <= black_pixel_count <= 67718 and 1243 <= len(contours) <= 1436:
        return 200, img
    elif 77774 <= black_pixel_count <= 87826 and 1238 <= len(contours) <= 1580:
        return 300, img
    elif 102196 <= black_pixel_count <= 110531 and 1409 <= len(contours) <= 1590:
        return 400, img
    else:
        return 0, img

def main():
    fry_count, img = counter()

    # SHOW THE RESULTS
    bg_img = init_bg(c=fry_count)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_cropped = crop_img(img_gray)
    img_inv = invert_img(img_cropped)
    diff_img = cv.absdiff(img_inv, bg_img)
    thresh = cv.adaptiveThreshold(invert_img(diff_img), 256, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31,5)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img_final = cv.drawContours(crop_img2(img), contours, -1, (0, 0, 2), 2)

    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', 600, 540)
    cv.putText(img_final, f'~Fry Count: {fry_count}', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow('image', img_final)
    cv.waitKey(0)
    cv.destroyAllWindows()

    main()

if __name__ == '__main__':
    main()