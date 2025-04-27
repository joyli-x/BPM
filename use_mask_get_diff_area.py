# 现有一张source_image和一张mask_image,mask_image中被修改过的区域为白色，其余为黑色，请将mask_image中被修改过的区域提取出来，保存为diff_area_image
import cv2
import numpy as np

source_image = cv2.imread('source_image.jpg')
mask_image = cv2.imread('mask_image.jpg', cv2.IMREAD_GRAYSCALE)

diff_area_image = cv2.bitwise_and(source_image, source_image, mask=mask_image)

cv2.imwrite('diff_area_image.jpg', diff_area_image)