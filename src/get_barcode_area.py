import os
import sys
import getopt

import cv2
from barcode_roi_locator import get_roi
import numpy as np
from logger import *


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
    a = 1.5
    enhanced = roi * float(a)
    enhanced[enhanced > 255] = 255
    enhanced = np.round(enhanced)
    enhanced = enhanced.astype(np.uint8)
    # cv2.imshow("对比度增强", enhanced)

    fixed_t = cv2.threshold(enhanced, 230, 255, cv2.THRESH_BINARY_INV)[1]  # 阈值可以低一些
    # cv2.imshow("thr", fixed_t)

    cleaned_ver = clean_ver(fixed_t)

    # 连接条形码
    element_link = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
    dilation_ed = cv2.dilate(cleaned_ver, element_link, iterations=1)  # 膨胀一次
    erosion_ed = cv2.erode(dilation_ed, element_link, iterations=1)  # 腐蚀一次
    # cv2.imshow("linked", erosion_ed)

    cleaned_hor = clean_hor(erosion_ed)
    # cv2.imshow("cleaned hor", cleaned_hor)

    biggest_rect = get_biggest_contour(cleaned_hor)
    brx1, bry1, brw, brh = biggest_rect  # 获得条形码的下边缘 （br = biggest rectangle）

    buttom_y = min(bry1 + brh + 40, h)
    # 绘制结果(条形码的底部开始)
    res = cv2.rectangle(rgb_roi, (min(brx1 + 15, int(w*0.20)), bry1+brh), (brx1 + brw - 10, buttom_y), (0, 128, 255), 2)
    #                                         左上顶点                              右下顶点
    return res


def resize_large(img):
    h, w, _ = img.shape
    # if w <= 1600:
    #     return img
    #  // 现在统一成1600宽度
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


def clean_ver(img):  # 消除细横线
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            curr_pixel = img[i][j]
            if curr_pixel == 0:
                continue
            i_f, j_f = i, j
            above, under = 0, 0
            #  纵向 （消除横线）
            while i_f < h - 1 and img[i_f][j] == 255:  # 往下走
                i_f += 1
                under += 1
                if under >= 9:
                    break
            i_f = i
            while i_f > 0 and img[i_f][j] == 255:  # 往上走
                i_f -= 1
                above += 1
                if above >= 9:
                    break
            if under + above < 9:  # 去掉粗细小于等于10的横线  （因为link_element的高度是10）
                for i_del in range(i - above, i + under + 1):
                    img[i_del][j] = 0
    return img


def clean_hor(img):  # 消除细竖线
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            curr_pixel = img[i][j]
            if curr_pixel == 0:
                continue
            i_f, j_f = i, j
            above, under, left, right = 0, 0, 0, 0
            #  横向 （消除竖线）
            while j_f < w-1 and img[i][j_f] == 255:  # 向右走
                j_f += 1
                right += 1
                if right >= 30:
                    break
            j_f = j
            while j_f > 0 and img[i][j_f] == 255:  # 向左走
                j_f -= 1
                left += 1
                if left >= 30:
                    break
            if left + right < 30:    # 去掉粗细小于等于30的竖线
                for j_del in range(j-left, j+right+1):
                    img[i][j_del] = 0
    return img


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
                p = p.split(".")[0] + ".png"
                out_file = os.path.join(dst_dir, p)
                cv2.imwrite(out_file, result_img)
                logger.info(p + " barcode area successful, No." + str(count))
            else:
                logger.warning("get barcode failed at: " + p)
        except Exception:
            logger.warning("Exception on: get_barcode_area.py-" + p)
            continue
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
        logger.info("%s %s", inputdir, outputdir)
        logger.error("invalid folder")


if __name__ == '__main__':
    main(sys.argv[1:])
