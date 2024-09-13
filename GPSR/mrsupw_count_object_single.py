#!/usr/bin/env python
# -*- coding: utf-8 -*-
# [centerX,centerY,width,height]
import argparse
import sys
import time

import cv2
import rospy
from geometry_msgs.msg import *
from pyk4a import PyK4A, ColorResolution,Config
from std_msgs.msg import *
from xm_msgs.msg import *
from xm_msgs.srv import *

sys.path.append("/home/xm/xm_vision/darknet")
import darknet

codeCodes = {
    'black': '0;30', 'bright gray': '0;37',
    'blue': '0;34', 'white': '1;37',
    'green': '0;32', 'bright blue': '1;34',
    'cyan': '0;36', 'bright green': '1;32',
    'red': '0;31', 'bright cyan': '1;36',
    'purple': '0;35', 'bright red': '1;31',
    '***': '0;33', 'bright purple': '1;35',
    'grey': '1;30', 'bright yellow': '1;33',
}


def colored(text, color='green'):
    return "\033[" + codeCodes[color] + "m" + text + "\033[0m"


def image_detection_original_no_image(image, network, class_names, thresh, fx, fy):
    global width, height
    darknet_image = darknet.make_image(width, height, 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    for i in range(len(detections)):
        detections[i] = list(detections[i])
        detections[i][2] = list(detections[i][2])
        detections[i][2][0] = (width / 2 / fx) - (1 / fx) * (width / 2 - detections[i][2][0]) if detections[i][2][
                                                                                                     0] <= width / 2 \
            else (1 / fx) * (detections[i][2][0] - width / 2) + (width / 2 / fx)
        detections[i][2][1] = (height / 2 / fy) - (1 / fy) * (height / 2 - detections[i][2][1]) if detections[i][2][
                                                                                                       1] <= height / 2 \
            else (1 / fy) * (detections[i][2][1] - height / 2) + (height / 2 / fy)
        detections[i][2][2] /= fx
        detections[i][2][3] /= fy
    darknet.free_image(darknet_image)
    return detections


def deal_with_desc(desc):
    global is_debug, types, class_names, type_classes
    is_type = False
    if desc in types:
        is_type = True
    count_num = 0
    try_count = 0
    while True:
        try_count += 1
        start_time = time.time()
        capture = camera.get_capture()
        detections = image_detection_original_no_image(capture.color, network, class_names,
                                                       0.7,
                                                       width / capture.color.shape[1],
                                                       height / capture.color.shape[0])
        if is_debug:
            drawn_image = darknet.draw_boxes(detections, capture.color, class_colors)
            cv2.putText(drawn_image, 'FPS: {:.2f}'.format((1 / (time.time() - start_time))), (20, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                        thickness=2)
            cv2.putText(drawn_image, 'Press \'Q\' to Exit!'.format((1 / (time.time() - start_time))), (20, 70),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                        thickness=2)
            cv2.imshow("Detected Image", drawn_image)
            if cv2.waitKey(1) in [ord("q"), ord("Q")]:
                cv2.destroyAllWindows()
                exit(0)
        for detection in detections:
            if (is_type and detection[0] in type_classes[desc]) or (is_type == False and detection[0] == desc):
                count_num += 1
        if count_num > 0:
            return count_num
        if try_count >= 100:
            return 0


def call_back(req):
    desc = req.des
    res = xm_count_objectResponse()
    num = deal_with_desc(desc)
    res.num = num
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', type=bool, default=False)
    args = parser.parse_args()
    is_debug = args.debug
    config_file = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/xm_vision.cfg'
    data_file = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/obj.data'
    weights = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/backup/xm_vision_last.weights'
    batch_size = 1
    types = ['food', 'drinks', 'toiletries']
    type_classes = {
        'food': ['cookie', 'gum'],
        'drinks': ['water', 'milk', 'cola', 'tea_pi', 'spring', 'juice'],
        'toiletries': ['soap', 'lotion', 'toothpaste']
    }
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=batch_size
    )
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    config = Config(color_resolution=ColorResolution.RES_1080P)
    camera = PyK4A(config)
    camera.start()
    first_capture = camera.get_capture()
    first_capture = camera.get_capture()
    fx = width / first_capture.color.shape[1]
    fy = height / first_capture.color.shape[0]
    rospy.init_node('countObjectNode')
    service = rospy.Service('countObject', xm_count_object, call_back)
    rospy.loginfo('GPSR Count Object\'s Vision Server Start!')
    rospy.spin()
