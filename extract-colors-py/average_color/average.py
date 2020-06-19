import numpy as np
import cv2

# image subtract on-nons
image_on = cv2.imread("C:\git\SmartBeauty\SmartBeauty\extract-colors-py\on.png", cv2.IMREAD_UNCHANGED)
image_non = cv2.imread("C:\git\SmartBeauty\SmartBeauty\extract-colors-py\non.png", cv2.IMREAD_UNCHANGED)
sub_img = cv2.subtract(image_on, image_non)
cv2.imshow("sub_img", sub_img)

# image extract


cv2.waitKey(0)
cv2.destroyAllWindows()