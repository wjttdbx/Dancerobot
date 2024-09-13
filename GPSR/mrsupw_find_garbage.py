#!/usr/bin/env python
# -*- coding: utf-8 -*-
# [centerX,centerY,width,height]
import argparse
import random
import sys
from datetime import datetime

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


def draw_boxes(detections, image, colors):
    for label, confidence, bbox in detections:
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color=colors[label], thickness=1,
                      lineType=cv2.LINE_AA)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (x - w // 2, y - h // 2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image


def transform_coordinate(coordinate):
    if coordinate[0] != -10:
        return [coordinate[2] / 1000, -coordinate[0] / 1000, -coordinate[1] / 1000]
    return coordinate


def get_closest_object(detections, transformed_image):
    if detections is not None and len(detections) > 0:
        detections.sort(key=lambda x: sum([v ** 2 for v in transformed_image[int(x[2][1])][int(x[2][0])]]))
        # return detections[0]
        for de in detections:
            cor = transformed_image[int(de[2][1])][int(de[2][0])]
            colored(f'{de[0]}:{cor}', 'bright cyan')
            if not (not cor[0] and not cor[1] and not cor[2]):
                return de
    return None


def get_type_of_object(detection):
    global type_classes
    for g_type in type_classes.keys():
        for name in type_classes[g_type]:
            if name == detection[0]:
                return g_type
    return None


def judge_cor(cor):
    if round(cor[0]) == round(cor[1]) == round(cor[2]) == 0:
        return False
    return True


def deal_with_request(req):
    global network, class_names, fx, fy
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d %Hh%Mm%Ss')
    count = 0
    while True:
        capture = camera.get_capture()
        detections = image_detection_original_no_image(capture.color, network, class_names, 0.8, fx, fy)
        drawn_image = draw_boxes(detections, capture.color, class_colors)
        cv2.imshow('debug', drawn_image)
        if cv2.waitKey(1) in [ord('q'), ord('Q')]:
            cv2.destroyAllWindows()
            exit(0)
        tar_detection = get_closest_object(detections, capture.transformed_depth_point_cloud)
        if not tar_detection:
            count += 1
            print(colored("Try {} times. No tar!".format(count), 'bright yellow'))
            if count >= 20:
                print(colored("There is no object!", 'bright red'))
                cv2.destroyAllWindows()
                return 'None', 'None', [-10, -10, -10]
            continue

        coordinate = capture.transformed_depth_point_cloud[int(tar_detection[2][1])][int(tar_detection[2][0])]
        print('coordinate', coordinate)
        c = 0
        while not judge_cor(coordinate):
            coordinate = capture.transformed_depth_point_cloud[int(tar_detection[2][1]) + random.randint(-10, 10)][
                int(tar_detection[2][0]) + random.randint(-10, 10)]
            c += 1
            if c >= 100:
                break
        if not judge_cor(coordinate):
            count += 1
            print(colored("Try {} times. No cor!".format(count), 'bright yellow'))
            if count >= 20:
                print(colored("There is a object! But I do not know the coordinate!", 'bright green'))
                cv2.destroyAllWindows()
                return get_type_of_object(tar_detection), tar_detection[0], [-1000.0, 0.0, 0.0]
            continue
        cv2.destroyAllWindows()
        cv2.imwrite(f'/home/xm/catkin_ws/src/xm_vision/src/scripts/GPSR/logs/{time_str}.jpg', drawn_image)
        return get_type_of_object(tar_detection), tar_detection[0], coordinate


def call_back(req):
    res = xm_find_objectResponse()
    o_type, name, coordinate = deal_with_request(req)
    print(o_type, name, coordinate)
    coordinate = transform_coordinate(coordinate)
    print(colored('type of object is {}'.format(o_type)))
    print(colored('name of object is {}'.format(name)))
    print(colored("transformed coordinate is {}".format(coordinate), "bright green"))
    res.type = o_type
    res.name = name
    if coordinate[0] != -10:
        coordinate[0] -= 0.2
    res.position.point.x, res.position.point.y, res.position.point.z = coordinate
    res.position.header.frame_id = "kinect2_rgb_link"
    res.position.header.stamp = rospy.Time(0)
    return res


if __name__ == "__main__":
    # True代表调试模式 False代表非调试模式
    parser = argparse.ArgumentParser()
    # True代表调试模式 False代表非调试模式
    parser.add_argument('-d', '--debug', type=bool, default=False)
    args = parser.parse_args()
    is_debug = args.debug

    type_classes = {
        'food': ['biscuit', 'chip', 'bread', 'cookie'],
        'drink': ['water', 'milk', 'sprite', 'cola', 'orange juice'],
        'clean': ['dish soap', 'hand wash', 'shampoo'],
    }

    # config_file = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/xm_vision.cfg'
    # data_file = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/obj.data_xmtest'
    # weights = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/backup/xm_vision_last.weights'
    # config_file = r'/home/xm/xm_vision/darknet/workspaces/xmtest/yolov4-custom-xmtest.cfg'
    # data_file = r'/home/xm/xm_vision/darknet/workspaces/xmtest/xmtest.data'
    # weights = r'/home/xm/xm_vision/darknet/workspaces/xmtest/yolov4-custom-xmtest_last.weights'
    # config_file = r'/home/xm/xm_vision/darknet/workspaces/xm/yolov4-custom-xm.cfg'
    # data_file = r'/home/xm/xm_vision/darknet/workspaces/xm/xm.data'
    # weights = r'/home/xm/xm_vision/darknet/workspaces/xm/yolov4-custom-xm_final.weights'
    # config_file = r'/home/xm/xm_vision/darknet/workspaces/xm-garbage/yolov4-custom-xm-garbage.cfg'
    # data_file = r'/home/xm/xm_vision/darknet/workspaces/xm-garbage/xm-garbage.data'
    # weights = r'/home/xm/xm_vision/darknet/workspaces/xm-garbage/yolov4-custom-xm-garbage-pre_19000.weights'
    config_file = r'/home/xm/xm_vision/darknet/workspaces/xm-garbage/yolov4-custom-xm-garbage.cfg'
    data_file = r'/home/xm/xm_vision/darknet/workspaces/xm-garbage/xm-garbage.data'
    weights = r'/home/xm/xm_vision/darknet/workspaces/xm-garbage/yolov4-custom-xm-garbage_7000.weights'
    batch_size = 1
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
    fx = width / first_capture.color.shape[1]
    fy = height / first_capture.color.shape[0]

    rospy.init_node('find_garbage_node')
    service = rospy.Service('find_garbage', xm_find_object, call_back)
    rospy.loginfo('find garbage service start!')
    rospy.spin()
