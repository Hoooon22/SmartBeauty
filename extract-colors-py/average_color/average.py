import numpy as np
import cv2

# image subtract on-nons
image_on = cv2.imread("on.png", cv2.IMREAD_UNCHANGED)
image_non = cv2.imread("non.png", cv2.IMREAD_UNCHANGED)
image_sub = cv2.add(image_on, 3)
cv2.imshow("image_sub", image_sub)

# image extract

cv2.waitKey(0)
cv2.destroyAllWindows()