# import time
import os
import sys
import getopt
import cv2
import numpy as np
from src.roi_locator import *


def get_CNY_area(filename):
    src = cv2.imread(filename)
    if src is None:
        return None
    ori_pic = src.copy()
    H, W, _ = ori_pic.shape  # 原图尺寸H x W
    src = resize_large(src)  # 太大的图片都缩小成1600的宽度
    h, w, _ = src.shape  # 压缩后尺寸 h x w
    # cv2.imshow('compressed', src)
    roi_x1, roi_y1, roi_x2, roi_y2 = get_roi(src)
    # roi_w = roi_x2 - roi_x1
    # roi_h = roi_y2 - roi_y1
    roi = cutROI(src.copy(), roi_x1, roi_y1, roi_x2, roi_y2)
    no_grain = del_background_grain(roi.copy())  # 去除后面的蓝色纹路，返回一个RGB图像的R通道
    thresholded = threshold_binary(no_grain.copy())  # 二值化

    element_size = 3
    without_lines = open_operation(thresholded, element_size)  # 大致去除横竖直线
    # cv2.imshow('first clean', without_lines)

    link_len = 70
    de_link_len = 60
    link_h = 9
    numbers_linked = link_numbers(without_lines, link_len, de_link_len, link_h)  # 把分离的数字都连在一起

    area_threshold = 1000 + (link_len - de_link_len) * link_h
    contours = get_contours(numbers_linked, area_threshold)

    #  绘制结果
    max_area = 0
    biggest_rect = None
    for cidx, c in enumerate(contours):
        # minAreaRect = cv2.minAreaRect(c)  # 连通区域的[最小]外接矩形, cv2.minAreaRect返回值：((cx, cy), (width, height), theta)
        (cx, cy, cw, ch) = cv2.boundingRect(c)  # 连通区域的[正]外接矩形， cv2.boundingRect的返回值：(x, y, w, h)
        if cw * ch > max_area:  # 如果现在这个比之前出现的更大
            biggest_rect = cv2.boundingRect(c)
            max_area = max(max_area, cw * ch)

    # 计算出未压缩图片的ROI
    ROI_X1, ROI_Y1 = int((roi_x1 / w) * W), int((roi_y1 / h) * H)  # 在未压缩图片中，ROI的左上顶点
    ROI_X2, ROI_Y2 = int((roi_x2 / w) * W), int((roi_y2 / h) * H)  # 在未压缩图片中，ROI的右下顶点
    ROI = ori_pic[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]  # 未压缩图片的ROI

    if biggest_rect is not None:
        (rx, ry, rw, rh) = biggest_rect
        offset = (link_len - de_link_len) // 2
        result_x1, result_y1 = rx + offset, ry  # 压缩图片roi中，结果的左上坐标
        result_x2, result_y2 = rx + rw - offset, ry + rh  # 压缩图片roi中，结果的右下坐标 TODO

        # 在未压缩的图ori_pic上绘制
        rate = (ROI_Y2 - ROI_Y1) / (roi_y2 - roi_y1)  # roi和ROI的尺寸比
        result_X1, result_Y1 = int(result_x1 * rate), int(result_y1 * rate)  # 未压缩图片ROI中，结果的左上
        result_X2, result_Y2 = int(result_x2 * rate), int(result_y2 * rate)  # 未压缩图片ROI中，结果的右下

        e = int(4*rate)  # 结果框离字符留一点小空间
        cv2.rectangle(ROI, (result_X1 -e, result_Y1 -e), (result_X2 +e, result_Y2 +e), (0, 128, 0), int(2*rate))

    return ROI  # 原图上的结果

def resize_large(img):
    h, w, _ = img.shape
    if w <= 1600:
        return img
    target_w = 1600
    target_h = int(h * target_w / w)
    de_size = (target_w, target_h)  # 注意：是先w再h，与以往不同
    little = cv2.resize(img, de_size, interpolation=cv2.INTER_AREA)
    #  双线性插值 cv2.INTER_LINEAR
    return little


#  去除背景蓝色细纹
def del_background_grain(img):
    h, w, _ = img.shape
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # RGB 转为 HSV
    lowerblue0 = np.array([78, 43, 46])  # 蓝色的 Hmin, Smin, Vmin
    upperblue0 = np.array([100, 255, 255])  # 蓝色的 Hmax, Smax, Vmax
    blue_is_255 = cv2.inRange(HSV, lowerblue0, upperblue0)  # 蓝的都为255(白)
    # blue_is_0 = 255 - blue_is_255  # 反过来 蓝的都为0（黑）
    for i in range(h):
        for j in range(w):
            b = blue_is_255[i][j]
            if b == 255:
                img[i][j] = [255, 255, 255]
    # cv2.imshow("blue_delted", img)
    return img


def getROI_fake(img):
    h, w, _ = img.shape
    x1 = int(w*0.70)
    y1 = int(h*0.67)
    x2 = int(w*0.90)
    y2 = int(h*0.78)
    return x1, y1, x2, y2


def cutROI(img, x1, y1, x2, y2):  # 左上角 和 右下角的坐标
    roi = img[y1: y2, x1: x2]
    return roi


# 通过阈值 进行二值化
def threshold_binary(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # ad_t_gaus = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
    # ad_t_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -2)
    fixed_t = cv2.threshold(img, 205, 255, cv2.THRESH_BINARY_INV)[1]
    return fixed_t

#  开操作，消除横竖线干扰
def open_operation(img, element_size):
    element_for_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (1, element_size))    #  处理横线
    erosion_ed = cv2.erode(img, element_for_hor, iterations=1)  # 腐蚀一次
    dilation_ed = cv2.dilate(erosion_ed, element_for_hor, iterations=1)  # 膨胀一次

    element_for_ver = cv2.getStructuringElement(cv2.MORPH_RECT, (element_size, 1))    # 处理竖线
    erosion_ed = cv2.erode(dilation_ed, element_for_ver, iterations=1)  # 腐蚀一次
    dilation_ed = cv2.dilate(erosion_ed, element_for_ver, iterations=1)  # 膨胀一次
    return dilation_ed


#  通过一个长方形的核，把分散的数字连接到一起
def link_numbers(img, link_len, de_link_len, link_h):
    link_element = cv2.getStructuringElement(cv2.MORPH_RECT, (link_len, link_h))
    numbers_linked = cv2.dilate(img, link_element, iterations=1)  # 通过一次膨胀连起来
    # cv2.imshow('number linked', numbers_linked)
    de_link_element = cv2.getStructuringElement(cv2.MORPH_RECT, (de_link_len, link_h))
    numbers_linked = cv2.erode(numbers_linked, de_link_element, iterations=1)  # 进行一次腐蚀
    # cv2.imshow('number linked eroded', numbers_linked)
    return numbers_linked



def get_contours(img, area_threshold):
    #  先消去过细的部分（没腐蚀掉的横竖线）
    img = clean_ver(img)
    img = clean_hor(img)

    h, w = img.shape
    # 消除不规则凸起
    # 1.统计最大宽度
    max_len = 0
    for i in range(h):  # 遍历每一行
        curr_len = 0
        for j in range(w):
            curr_pixel = img[i][j]
            if curr_pixel == 255:
                curr_len += 1
            if curr_pixel == 0:
                max_len = max(curr_len, max_len)
                curr_len = 0

    # 2.将小于最大宽度80%的白色横线整条删去
    len_thres = int(max_len * 0.8)
    for i in range(h):  # 遍历每一行
        start, end = 0, 0
        c_len = 0
        for j in range(w):
            curr_pixel = img[i][j]
            if curr_pixel == 0 or j == w-1:  # 碰到0或者走到尽头了
                if j == w-1:
                    end += 1
                c_len = end - start
                if c_len != 0 and c_len < len_thres:
                    for j_del in range(start, end+1):  # 消除
                        img[i][j_del] = 0
                start = j
                end = j
            if curr_pixel == 255:
                end += 1

    # 再消除一次细横线（因为消除手枪把之后可能会出现新的横线）
    img = clean_ver(img)

    #                                                  􏱚   轮廓检索模式􏰎              轮廓近似方法
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 剔除一些面积比较小的连通域
    big_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if abs(area) > area_threshold:  # 面积小于阈值的就无视
            big_contours.append(c)
    # rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    # cv2.drawContours(rgb_img, big_contours, -1, (0, 0, 255), 1)  # 在边缘画线
    # cv2.imshow('with contours', rgb_img)
    return big_contours

def clean_ver(img):  # 消除细横线
    h, w = img.shape
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
                if under > 10:
                    continue
            i_f = i
            while i_f > 0 and img[i_f][j] == 255:  # 往上走
                i_f -= 1
                above += 1
                if above > 10:
                    continue
            if under + above < 9:  # 去掉粗细小于等于10的横线  （因为link_element的高度是10）
                for i_del in range(i - above, i + under + 1):
                    img[i_del][j] = 0
    return img

def clean_hor(img):  # 消除细竖线
    h, w = img.shape
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
                if right > 8:
                    continue
            j_f = j
            while j_f > 0 and img[i][j_f] == 255:  # 向左走
                j_f -= 1
                left += 1
                if left > 8:
                    continue
            if left + right < 7:    # 去掉粗细小于等于7的竖线（像素左右各有3个点）
                for j_del in range(j-left, j+right+1):
                    img[i][j_del] = 0
    return img



def process(src_dir, dst_dir):
    count = 0
    p_list = os.listdir(src_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for p in p_list:
        print("getting CNY area:", p)
        try:
            file = os.path.join(src_dir, p)
            result_img = get_CNY_area(file)
            if result_img is not None:
                out_file = os.path.join(dst_dir, p)
                cv2.imwrite(out_file, result_img)
                print(count)
            else:
                print("get CNY failed at:", p)
        except Exception:
            print("Exception on: get_CNY_area.py --", p)
            sys.exit(3)
        finally:
            count += 1
    print("Finished: " + str(count) + " CNY areas")


def main(argv):
    inputdir = ''
    outputdir = ''
    try:
        opts, args = getopt.getopt(argv, "h", ["indir=", "outdir="])
    except getopt.GetoptError:
        print('get_CNY_area.py --indir <inputdir> --outdir <outputdir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "--indir":
            inputdir = arg
        elif opt == "--outdir":
            outputdir = arg
    if os.path.isdir(inputdir) and os.path.isdir(outputdir):
        process(inputdir, outputdir)
    else:
        print("invalid folder")


if __name__ == '__main__':
    main(sys.argv[1:])