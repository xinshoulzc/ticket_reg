import os
import numpy as np
import cv2 as cv
import copy as cp
import sys, getopt
# import shutil

DEBUG = 0

MODE_PRICE = 1
MODE_BARCODE = 2

MARK_COLOR_PRICE = np.array([0, 128, 0])
MARK_COLOR_BCODE = np.array([0, 128, 255])

# remove line process direction
DIR_TOP_DOWN = 0
DIR_BOTTOM_UP = 1

def add_suffix(filename, suffix):
    names = filename.split(".")
    return names[0] + "_" + suffix + "." + names[-1]


def rmv_suffix(filename, cnt):
    names = filename.split(".")
    parts = names[0].split("_")
    size = len(parts)
    if cnt > size - 1:
        return filename
    parts = parts[0:size - cnt]

    ret = '_'.join(parts)
    return ret + "." + names[-1]


# return binary image
def show_img(img):
    cv.imshow("a", img)
    cv.waitKey()


# binary
# 用于确定二值化的最小亮度阈值，亮度阈值x是满足:
# img[v < x] >= MIN_BLACK_COUNT 的最小值，v为像素亮度
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


# 去除横线
def try_rmv_line(img, row_idx, col_begin, col_end, direction):
    i = row_idx
    h, w = img.shape[:2]

    if i == 0 or i == 1:
        img[i, col_begin:col_end] = 255
    elif i == h - 1 or i == h - 2:
        img[i, col_begin:col_end] = 255
    else:
        for j in range(col_begin, col_end):
            if direction == DIR_TOP_DOWN:
                if img[i - 1][j] == 255:
                    img[i][j] = 255
                elif img[i - 2][j] == 255:
                    img[i - 1:i + 1, j] = 255
            else:
                if img[i + 1][j] == 255:
                    img[i][j] = 255
                elif img[i + 2][j] == 255:
                    img[i:i + 2, j] = 255


def check_line(img, row, direction, min_line_length):
    h, w = img.shape[:2]
    connect_count = 0
    line_begin = -1
    for j in range(w):
        if img[row, j] == 0:
            if line_begin == -1:
                line_begin = j
            connect_count += 1

            if j == w - 1 and connect_count >= min_line_length:
                try_rmv_line(img, row, line_begin, w, direction)
        else:
            if connect_count >= min_line_length:
                try_rmv_line(img, row, line_begin, j, direction)
            line_begin = -1
            connect_count = 0


def rmv_line(img, min_line_length):
    h, w = img.shape[:2]
    tmp = cp.copy(img)
    # top-down
    for i in range(h):
        check_line(img, i, DIR_TOP_DOWN, min_line_length)
    # bottom-up
    for i in range(h - 1, -1, -1):
        check_line(tmp, i, DIR_BOTTOM_UP, min_line_length)
    img = cv.bitwise_and(img, tmp)
    return img


def filter_noise_by_area(img, min_area, connectivity):
    reversed_img = ~img
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(reversed_img, connectivity=connectivity)
    # the following part used to remove background component
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    for i in range(nb_components):
        if sizes[i] < min_area:
            img[output == i + 1] = 255


class DigitZone:
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.center = (left + right) // 2
        self.width = right - left

    def merge(self, other):
        if not isinstance(other, DigitZone):
            return NotImplemented
        left = min(self.left, other.left)
        right = max(self.right, other.right)
        return DigitZone(left, right)

    def split(self, amount):
        if amount == 0:
            return [self]
        zones = []
        width = self.width // amount
        cur = self.left
        while cur + width <= self.right:
            zones.append(DigitZone(cur, cur + width))
            cur += width
        # last one
        if cur < self.right:
            zones[-1] = zones[-1].merge(DigitZone(cur, self.right))
        return zones


# separate digits
# if black pixels in one column <= BREAK_THRESHOLD: do break
def find_digit_zone(img, break_threshold):
    h, w = img.shape[:2]
    digit_zones = []
    begin = -1
    for j in range(w):
        # cut last digit
        if j == w - 1 and begin != -1:
            digit_zones.append(DigitZone(begin, w))
            break

        px_num = 0
        for i in range(h):
            if img[i][j] == 0:
                px_num += 1

        if px_num > break_threshold:
            if begin == -1:
                begin = j
        else:
            if begin != -1:
                end = j
                digit_zones.append(DigitZone(begin, end))
                begin = -1
    return digit_zones


def separate_digit(img, break_threshold, fix_char_width, min_char_width, max_char_width, min_char_gap,
                   min_char_height):
    h, w = img.shape[:2]
    digit_zones = find_digit_zone(img, break_threshold)
    i = 0
    # merge
    while i < len(digit_zones) - 1:
        zone = digit_zones[i]
        while zone.width < min_char_width:
            next_zone = digit_zones[i + 1]
            if zone.width + next_zone.width < max_char_width and next_zone.left - zone.right < min_char_gap:
                zone = zone.merge(next_zone)
                digit_zones[i] = zone
                del digit_zones[i + 1]
                if len(digit_zones) - 1 == i:
                    break
            else:
                break
        i += 1

    # split
    i = 0
    while i < len(digit_zones):
        zone = digit_zones[i]
        if zone.width > max_char_width:
            zones = zone.split(zone.width // fix_char_width)
            del digit_zones[i]
            for j in range(len(zones) - 1, -1, -1):
                digit_zones.insert(i, zones[j])
        i += 1

    # filter dots
    digit_imgs = []
    count = 0
    for zone in digit_zones:
        digit_img = img[:, zone.left:zone.right]

        trim_height = 0
        for i in range(h):
            if (digit_img[i] == 0).sum() != 0:
                trim_height += 1

        if trim_height > min_char_height:
            digit_imgs.append(~digit_img)
            count += 1

    return digit_imgs


def get_roi_img(img, mark_color):
    mask = cv.inRange(img, mark_color, mark_color)
    c, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(c)):
        if hierarchy[0][i][3] >= 0:
            xs = [point[0][0] for point in c[i]]
            ys = [point[0][1] for point in c[i]]
            x_low = np.min(xs)
            x_high = np.max(xs)
            y_low = np.min(ys)
            y_high = np.max(ys)

            if y_low +2 >= y_high or x_low+2 >= x_high:
                return None
            dstImg = img[y_low + 2:y_high, x_low + 2:x_high]
            return dstImg


def roi_to_digit_img(img, mode):
    # resize
    Y_SIZE = 2*img.shape[0]
    X_SIZE = 2*img.shape[1]
    img = cv.resize(img, (X_SIZE, Y_SIZE))


    min_black_count = 0
    break_threshold = 0
    if mode == MODE_PRICE:
        min_black_count = 0.35 * X_SIZE * Y_SIZE
        break_threshold = 1
    elif mode == MODE_BARCODE:
        min_black_count = 0.2 * X_SIZE * Y_SIZE
        break_threshold = 3

    min_line_length = 35
    min_area = (Y_SIZE / 6) ** 2
    fixed_char_width = 30  # ~35
    min_char_width = 4 * fixed_char_width / 5
    min_char_gap = fixed_char_width / 3
    max_char_width = 8 * fixed_char_width / 5
    min_char_height = Y_SIZE / 2

    if mode == MODE_PRICE:
        img = binary(img, min_black_count)
        # remove line: simply count horizontal pixels
        img = rmv_line(img, min_line_length)
        # filter bigger noise
        filter_noise_by_area(img, min_area, 8)

    elif mode == MODE_BARCODE:
        img = binary(img, min_black_count)

    # save all chars' image
    digit_imgs = separate_digit(img, break_threshold, fixed_char_width, min_char_width, min_char_gap, max_char_width,
                                min_char_height)
    return digit_imgs

# mode : 1:price; 2:barcode
def process(src_dir, dst_dir, mode):
    mark_color = MARK_COLOR_PRICE
    if mode == MODE_BARCODE:
        mark_color = MARK_COLOR_BCODE

    entries = os.listdir(src_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    # else:
    #     shutil.rmtree(dst_dir)
    #     os.mkdir(dst_dir)
    for entry in entries:
        file = os.path.join(src_dir, entry)
        if os.path.isfile(file):
            img = cv.imread(file)
            if img is None:
                print("Open image 「" + file + "」failed.")
                continue
            roi = get_roi_img(img, mark_color)
            if roi is None:
                print("Image 「"+ file + "」 has no roi area.")
                continue
            digit_imgs = roi_to_digit_img(roi, mode)
            for i, digit in enumerate(digit_imgs):
                cv.imwrite(os.path.join(dst_dir, add_suffix(entry, str(i))), digit)

def main(argv):
    inputdir = ''
    outputdir = ''
    mode = 1
    try:
        opts, args = getopt.getopt(argv, "h", ["indir=", "outdir=", "mode="])
        for opt, arg in opts:
            if opt == "--indir":
                inputdir = arg
            elif opt == "--outdir":
                outputdir = arg
            elif opt == "--mode":
                mode = arg
        process(inputdir, outputdir, mode)
    except getopt.GetoptError:
        print('Usage: digit_segment.py --indir <inputdir> --outdir <outputdir> --mode <mode>')
        sys.exit(-1)
    except Exception as e:
        print('Error: ', e)
        sys.exit(-2)


if __name__ == "__main__":
    main(sys.argv[1:])
    # process('../datasets/muti_digit','../datasets/single_digit', 2)
