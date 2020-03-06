import cv2 as cv
import numpy as np
import math

# max_white_w: 单行最大白色像素数，一行中白色像素数量超过此值则认为是白边
# max_white_h: 单列最大白色像素数
def rmv_bound(img, max_white_w, max_white_h):
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(hsv)
    (h, w) = img.shape[:2]

    top = left = 0
    bottom = h
    right = w
    for i in range(h):
        if (v[i] == 255).sum() < max_white_w:
            top = i
            break
    for i in range(h - 1, -1, -1):
        if (v[i] == 255).sum() < max_white_w:
            bottom = i + 1
            break
    for i in range(w):
        if (v[:, i] == 255).sum() < max_white_h:
            left = i
            break
    for i in range(w - 1, -1, -1):
        if (v[:, i] == 255).sum() < max_white_h:
            right = i + 1
            break
    return left, top, img[top:bottom, left:right]

def binary(img, min_black_count):
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(hsv)

    min_v = 0
    black_count = 0
    for i in range(256):
        black_count += (v == i).sum()
        if black_count >= min_black_count:
            min_v = i
            break

    v[v <= min_v] = 0
    v[v > min_v] = 255

    return v

def filter_noise_by_whRatio(img, min_w_h_ratio):
    cnts = cv.findContours(img, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE)[0]
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        if w / h < min_w_h_ratio:
            cv.drawContours(img, [c], 0, 0, -1)
            continue

def draw_lines(img, min_line_len, max_line_gap, slope_threshold):
    line_back = np.zeros(img.shape).astype(np.uint8)
    lines = cv.HoughLinesP(img, 1.0, np.pi / 180, 150, minLineLength=min_line_len, maxLineGap=max_line_gap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = 1 if x2 - x1 == 0 else (y2 - y1) / (x2 - x1)
            if abs(k) < slope_threshold:
                cv.line(line_back, (x1, y1), (x2, y2), 255)

    return line_back

class box:
    def __init__(self, x, y, w, h, contour):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.contour = contour

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

def fix_tilt(img, k):
    h, w = img.shape[:2]
    # 1
    for y in range(h):
        for x in range(w):
            # dist: y - y0, y为当前点y值，y0为直线上横坐标取x时对应的y值
            if k < 0:
                top_dist = y - k * (x - w + 1)
                bottom_dist = y - (h - 1 + k * x)
            else:
                top_dist = y - k * x
                bottom_dist = y - (h - 1 + k * (x - w + 1))

            if top_dist < 0 or bottom_dist > 0:
                img[y][x] = (0, 0, 0)
    # 2
    input_img = rotateImage(img, math.degrees(math.atan(k)))
    # 3
    top = 0
    bottom = h - 1
    left = 0
    right = w - 1
    for y in range(h):
        if np.any(input_img[y] > [10, 10, 10]):
            top = y
            break
    for y in range(h - 1, -1, -1):
        if np.any(input_img[y] > [10, 10, 10]):
            bottom = y + 1
            break
    for x in range(w):
        if np.any(input_img[:, x] > [10, 10, 10]):
            left = x
            break
    for x in range(w - 1, -1, -1):
        if np.any(input_img[:, x] > [10, 10, 10]):
            right = x + 1
            break
    return input_img[top:bottom, left:right]


def find_valid_area(input_img, line_img):
    cnts = cv.findContours(line_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)[0]

    box_sorted = []
    # loop over the contours
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        if len(box_sorted) == 0:
            box_sorted.append(box(x, y, w, h, c))
            continue

        for i in range(len(box_sorted)):
            if y < box_sorted[i].y:
                box_sorted.insert(i, box(x, y, w, h, c))
                break

    # 找到合适的线条区域，处理没找到足够直线的情况
    h, w = input_img.shape[:2]
    top_line = None
    bottom_line = None
    line_cnts = len(box_sorted)
    if line_cnts < 2:
        # 没找到直线，直接返回
        # return input_img[int(h / 2):h, int(w / 2):w]
        return int(h/2), h-1
    elif line_cnts < 7:
        # 没找齐
        line_gap = box_sorted[-1].y - box_sorted[-2].y
        for i in range(line_cnts - 3, -1, -1):
            new_gap = box_sorted[i + 1].y - box_sorted[i].y
            if new_gap > 3 * line_gap:
                top_line = box_sorted[i + 1]
                bottom_line = box_sorted[i + 3] if i + 3 < line_cnts else box_sorted[-1]
            line_gap = new_gap

        if top_line is None:
            return int(h / 2), h - 1
            # return input_img[int(h / 2):h, int(w / 2):w]
    else:
        top_line = box_sorted[3]
        bottom_line = box_sorted[5]

    [vx, vy, xt, yt] = cv.fitLine(top_line.contour, cv.DIST_L2, 0, 0.01, 0.01)
    k1 = vy / vx
    [vx, vy, xb, yb] = cv.fitLine(bottom_line.contour, cv.DIST_L2, 0, 0.01, 0.01)
    k2 = vy / vx
    k = (k1 + k2) / 2

    # 截取区域确定
    h, w = input_img.shape[:2]
    if k < 0:
        # 斜线右高(因为y轴是反的)
        top = yt + k * (w - 1 - xt)
        bottom = yb + k * (w / 2 - xb)
    else:
        top = yt + k * (w / 2 - xt)
        bottom = yb + k * (w - 1 - xb)



    input_img = input_img[int(top):int(bottom), int(w / 2):w]
    # if abs(k) > 0.01:
    #     input_img = fix_tilt(input_img, k)

    gap = (bottom - top)/2
    top_padding = 0.4 * gap
    bottom_padding = 0.3 * gap
    top = top + top_padding
    bottom = bottom - bottom_padding

    return int(top), int(bottom)

def get_roi(img):
    # 去除白边
    h, w = img.shape[:2]
    left_padding, top_padding, img = rmv_bound(img, 0.9*w, 0.9*h)
    # 裁剪减少计算量，裁剪后的部分不能太小，以确保待检测直线有足够大的长宽比，与杂质区分开
    w_before_crop = img.shape[1]
    cropped_img = img[:, w_before_crop // 3:w_before_crop]
    h, w = cropped_img.shape[:2]
    # start modify pixels
    # 去掉一些图中的黑边
    mask = cv.inRange(cropped_img, (0, 0, 0), (20, 20, 20))
    mask3 = cv.merge((mask, mask, mask))
    img = cv.bitwise_or(cropped_img, mask3)
    # 二值化
    img = ~binary(img, 0.22 * h * w)
    # 开操作，用横向结构元去掉竖线
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (26, 1))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1)
    # 膨胀，用圆形结构元将非直线的剩余杂质连起来
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 5))
    img = cv.morphologyEx(img, cv.MORPH_DILATE, kernel, iterations=1)
    # 按宽高比去掉杂质
    filter_noise_by_whRatio(img, 18)

    # 查找直线
    line_back = draw_lines(img, 0.5 * w, 0.4 * w, 0.05)
    # 膨胀，合并相近的直线
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 10))
    line_back = cv.morphologyEx(line_back, cv.MORPH_DILATE, kernel, iterations=1)
    # 提取第4到第6根直线区域
    top, bottom = find_valid_area(cropped_img, line_back)

    return int(left_padding+w_before_crop*0.7), top + top_padding, int(left_padding+w_before_crop*0.95), bottom + top_padding

