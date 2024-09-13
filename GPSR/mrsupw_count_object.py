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

#指定一张图片的位置/或使用cv2.imread() 的结果，进行model预测+画框+另存为新图片
#huanhuatupiandaxiao
def image_detection_original_no_image(image, network, class_names, thresh, fx, fy, width, height):
    darknet_image = darknet.make_image(width, height, 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    #
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


def filter_detections(detections):
    new_detections = []
    for detection in detections:
        if detection[0] in ['apple', 'orange']:
            new_detections.append(detection)
    return new_detections


def deal_with_desc(desc):
    global is_debug, types, class_names1, class_names2, type_classes, width1, width2, height1, height2
    is_type = False
    if desc in types:
        is_type = True
    count_num = 0
    try_count = 0
    while True:
        try_count += 1
        #lushijain
        start_time = time.time()
        capture = camera.get_capture()
        detections1 = image_detection_original_no_image(capture.color, network1, class_names1,
                                                        0.7,
                                                        width1 / capture.color.shape[1],
                                                        height1 / capture.color.shape[0], width1, height1)
        detections2 = filter_detections(image_detection_original_no_image(capture.color, network2, class_names2,
                                                                          0.7,
                                                                          width2 / capture.color.shape[1],
                                                                          height2 / capture.color.shape[0], width2, height2))

        detections = detections1 + detections2
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
    types = ['drinks', 'snacks', 'food', 'fruits']
    type_classes = {
        'food': ['noodles', 'sausage'],
        'drinks': ['chocolate drink', 'milk', 'sprite', 'coke', 'grape juice', 'orange juice'],
        'snacks': ['pringle', 'cracker', 'potato chips'],
        'fruits': ['apple', 'orange']
    }

    config_file1 = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/xm_vision.cfg'
    data_file1 = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/obj.data_xmtest'
    weights1 = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/backup/xm_vision_last.weights'
    batch_size1 = 1

    network1, class_names1, class_colors1 = darknet.load_network(
        config_file1,
        data_file1,
        weights1,
        batch_size=batch_size1
    )
    width1 = darknet.network_width(network1)
    height1 = darknet.network_height(network1)

    config_file2 = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/xm_vision.cfg'
    data_file2 = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/obj.data'
    weights2 = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/backup/xm_vision_last.weights'
    batch_size2 = 1
    network2, class_names2, class_colors2 = darknet.load_network(
        config_file2,
        data_file2,
        weights2,
        batch_size=batch_size2
    )
    width2 = darknet.network_width(network2)
    height2 = darknet.network_height(network2)

    config = Config(color_resolution=ColorResolution.RES_1080P)
    camera = PyK4A(config)
    camera.start()
    first_capture = camera.get_capture()
    fx1 = width1 / first_capture.color.shape[1]
    fy1 = height1 / first_capture.color.shape[0]
    fx2 = width2 / first_capture.color.shape[1]
    fy2 = height2 / first_capture.color.shape[0]

    rospy.init_node('countObjectNode')
    service = rospy.Service('countObject', xm_count_object, call_back)
    rospy.loginfo('GPSR Count Object\'s Vision Server Start!')
    rospy.spin()
