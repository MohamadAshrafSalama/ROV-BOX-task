import cv2
import numpy as np
saved = []
key = cv2.waitKey(1)
webcam = cv2.VideoCapture(1)

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)
def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)
def crop(image):
    edged = cv2.Canny(image, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cnts= cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ROI = image[y:h, x:w]
        cv2.imshow("ROI",ROI)
        return ROI

while webcam.isOpened():
        check, frame = webcam.read()
        if check:
            cv2.imshow("Capturing", frame)
            if cv2.waitKey(1) & 0xFF == ord('a'):
                cv2.imwrite(filename='task1.jpg', img=frame)

                if ord('a'):
                    print("saved image 1")

            elif cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite(filename='task2.jpg', img=frame)
                if ord('s'):
                    print("saved image 2")

            elif cv2.waitKey(1) & 0xFF == ord('d'):
                 cv2.imwrite(filename='task3.jpg', img=frame)
                 if ord('d'):
                     print("saved image 3")

            elif cv2.waitKey(1) & 0xFF == ord('f'):
                 cv2.imwrite(filename='task4.jpg', img=frame)
                 if ord('f'):
                     print("saved image 4")

            elif cv2.waitKey(1) & 0xFF == ord('g'):
                 cv2.imwrite(filename='task5.jpg', img=frame)
                 if ord('g'):
                     print("saved image 5")


            elif cv2.waitKey(1) & 0xFF == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
                

while True:
    img1 = cv2.imread("s4.jpg")
    img1_crop = crop(img1)
    img2 = cv2.imread("s2.jpg")
    img2_crop = crop(img2)
    img3 = cv2.imread("s1.jpg")
    img3_crop = crop(img3)
    img4 = cv2.imread("s3.jpg")
    img4_crop = crop(img4)
    img5 = cv2.imread("s5.jpg")
    img5_crop = crop(img5)
    blank_image2 = 255 * np.ones(shape=[500, 550, 3], dtype=np.uint8)
    #blank_image2 = 255 * np.zeros((500, 500, 3), np.uint8)
    im_tile_resize = concat_tile_resize([
        [blank_image2, img3_crop, blank_image2, blank_image2, blank_image2],
        [img1_crop, img2_crop, img4_crop, img5_crop]])
    cv2.imshow("Box",im_tile_resize)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break