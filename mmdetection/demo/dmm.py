import asyncio
from argparse import ArgumentParser
import cv2
import numpy as np
from webcolors import rgb_to_name
import webcolors
import mmcv
from sklearn.cluster import KMeans
from scipy.spatial import distance

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    args = parser.parse_args()
    return args

def get_average_color(img, bbox):
    if img is None:
        return None
    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    roi_flat = roi.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(roi_flat)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return dominant_color.astype(int)

def closest_color(request_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - request_color[0]) ** 2
        gd = (g_c - request_color[1]) ** 2
        bd = (b_c - request_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def main(args):
    image = mmcv.imread(args.img)
    bbox = (24, 38, 357, 192)
    average_color = get_average_color(image, bbox)
    # r_av, g_av, b_av = average_color
    # average_color = [255-r_av, 255-g_av, 255-b_av]
    closest_html_color = closest_color(average_color)

    print("Average Color is: ", average_color)
    print("HTML Color is: ", closest_html_color)


if __name__ == '__main__':
    args = parse_args()
    main(args)

# image = cv2.imread("https://github.com/MroutsideturnAF/FashionFormer/raw/main/figs/sample_image.png?raw=true") #.imread('girl.png')
# image = mmcv.imread("demo.jpg")
# print(image)