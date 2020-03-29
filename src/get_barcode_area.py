# from datetime import datetime
import os
import sys
import getopt

import cv2
from src.barcode_roi_locator import *
from src.logger import *

def get_barcode_area(filename):
    src = cv2.imread(filename)
    if src is None:
        return None
    src = resize_large(src)
    roi_x1, roi_y1, roi_x2, roi_y2 = get_roi(src)
    w = roi_x2 - roi_x1
    h = roi_y2 - roi_y1
    roi = cutROI(src, roi_x1, roi_y1, roi_x2, roi_y2 + 5)
    rgb_roi = roi.copy()
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    fixed_t = cv2.threshold(roi, 160, 255, cv2.THRESH_BINARY_INV)[1]  # 阈值可以低一些

    element_link = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    dilation_ed = cv2.dilate(fixed_t, element_link, iterations=1)  # 膨胀一次
    erosion_ed = cv2.erode(dilation_ed, element_link, iterations=1)  # 腐蚀一次

    biggest_rect = get_biggest_contour(erosion_ed)
    brx1, bry1, brw, brh = biggest_rect  # 获得条形码的下边缘 （br = biggest rectangle）

    # 绘制结果(条形码的底部开始)
    res = cv2.rectangle(rgb_roi, (min(brx1 + 5, int(w*0.20)), bry1+brh), (brx1 + brw - 5, h), (0, 128, 255), 2)

    return res

def resize_large(img):
    h, w, _ = img.shape
    if w <= 1600:
        return img
    target_w = 1600
    target_h = int(h * target_w / w)
    de_size = (target_w, target_h)  # 注意：是先w再h，与以往不同
    little = cv2.resize(img, de_size, interpolation=cv2.INTER_AREA)
    return little


def get_biggest_contour(img):
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 先剔除一些面积比较小的连通域
    big_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if abs(area) > 500:  # 面积小于阈值的就无视
            big_contours.append(c)
    max_area = 0
    biggest_rect = None
    for cidx, c in enumerate(big_contours):
        (cx, cy, cw, ch) = cv2.boundingRect(c)  # 连通区域的[正]外接矩形， cv2.boundingRect的返回值：(x, y, w, h)
        if cw * ch > max_area:  # 如果现在这个比之前出现的更大
            biggest_rect = cv2.boundingRect(c)
            max_area = max(max_area, cw * ch)
    return biggest_rect

def cutROI(img, x1, y1, x2, y2):  # 左上角 和 右下角的坐标
    roi = img[y1: y2, x1: x2]
    return roi



# if __name__ == '__main__':
#     file = 'all/01070.png'
#     result = get_barcode_area(file)
#     if result is not None:
#         cv2.imshow('BarCode '+': '+file, result)
#         while True:
#             # 键盘检测函数，0xFF是因为64位机器
#             k = cv2.waitKey(1) & 0xFF
#             if k == ord('q'):
#                 break
#         cv2.destroyAllWindows()


def process(src_dir, dst_dir):
    count = 0
    p_list = os.listdir(src_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for p in p_list:
        logger.info("start getting barcode area: " + p)
        try:
            file = os.path.join(src_dir, p)
            result_img = get_barcode_area(file)
            if result_img is not None:
                out_file = os.path.join(dst_dir, p)
                cv2.imwrite(out_file, result_img)
                logger.info(p + " barcode area successful, No." + str(count))
            else:
                logger.warning("get barcode failed at: " + p)
        except Exception:
            logger.warning("Exception on: get_barcode_area.py-" + p)
            sys.exit(3)
        finally:
            count += 1
    logger.info("Finished: got " + str(count) + " barcode areas")


def main(argv):
    inputdir = ''
    outputdir = ''
    try:
        opts, args = getopt.getopt(argv, "h", ["indir=", "outdir="])
    except getopt.GetoptError:
        logger.error('getOptError: get_barcode_area.py --indir <inputdir> --outdir <outputdir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "--indir":
            inputdir = arg
        elif opt == "--outdir":
            outputdir = arg
    if os.path.isdir(inputdir) and os.path.isdir(outputdir):
        process(inputdir, outputdir)
    else:
        logger.error("invalid folder")


if __name__ == '__main__':
    main(sys.argv[1:])
