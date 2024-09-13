#!/usr/bin/env python
# -*- coding: utf-8 -*-
# [centerX,centerY,width,height]
import argparse
import sys
import time

import cv2
import rospy
from geometry_msgs.msg import *
from pyk4a import PyK4A, ColorResolution, Config
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


def image_detection_original_no_image(image, network, class_names, thresh, fx, fy, width, height):
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


def filter_detections(detections):
    new_detections = []
    for detection in detections:
        if detection[0] in ['apple', 'orange']:
            new_detections.append(detection)
    return new_detections


def deal_with_req(name, type_name, adj):
    global is_debug, types, class_names1, class_names2, type_classes, width1, width2, height1, height2
    if name != '':
        # 说明要找一个物体 这不简单？
        try_count = 0
        while True:
            try_count += 1
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
                if detection[0] == name:
                    # 找到了 就这？
                    coordinate = capture.transformed_depth_point_cloud[int(detection[2][1]), int(detection[2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
            if try_count >= 100:
                return '', [-10, -10, -10]
    elif type_name != '' and adj == '':
        # 说明要找某一类的东西
        # 说明要找一个物体 这不简单？
        objects = []
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
                if detection[0] in type_classes[type_name]:
                    objects.append(detection)
            if len(objects) > 0:
                # 说明找到了 就这？
                name = ','.join([v[0] for v in objects])
                print(colored("name:{}".format(name)))
                coordinate = capture.transformed_depth_point_cloud[int(objects[len(objects) // 2][2][1]), int(objects[len(objects) // 2][2][0])]
                if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                    # 说明坐标有效
                    coordinate = [v / 1000 for v in coordinate]
                return name, coordinate
            if try_count >= 100:
                return '', [-10, -10, -10]

    elif type_name == '' and adj != '':
        if adj in ['biggest', 'largest']:
            # 说明要在全部物体中找到最大的 这不简单？
            objects = []
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
                for name in size_classes:
                    for detection in detections:
                        if detection[0] == name:
                            objects.append(detection)
                            break
                    if len(objects) >= 1:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[0][2][1]), int(objects[0][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]
        if adj in ['smallest', 'thinnest']:
            # 说明要在全部物体中找到最小的 这不简单？
            objects = []
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
                for name in size_classes[::-1]:
                    for detection in detections:
                        if detection[0] == name:
                            objects.append(detection)
                            break
                    if len(objects) >= 3:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[0][2][1]), int(objects[0][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]

        if adj == 'heaviest':
            # 说明要在全部物体中找到最重的 这不简单？
            objects = []
            try_count = 0
            while True:
                try_count += 1
                start_time = time.time()
                capture = camera.get_capture()
                detections = image_detection_original_no_image(capture.color, network, class_names,
                                                               0.7,
                                                               width / capture.color.shape[1],
                                                               height / capture.color.shape[0],width,height)
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
                for name in weight_classes:
                    for detection in detections:
                        if detection[0] == name:
                            objects.append(detection)
                            break
                    if len(objects) >= 1:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[0][2][1]), int(objects[0][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]
        if adj == 'lightest':
            # 说明要在全部物体中找到最轻的 这不简单？
            objects = []
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
                for name in weight_classes[::-1]:
                    for detection in detections:
                        if detection[0] == name:
                            objects.append(detection)
                            break
                    if len(objects) >= 3:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[0][2][1]), int(objects[0][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]

        if adj in ['three_biggest', 'three_largest']:
            # 说明要在全部物体中找到前三大的 这不简单？
            objects = []
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
                for name in size_classes:
                    for detection in detections:
                        if detection[0] == name:
                            objects.append(detection)
                            break
                    if len(objects) >= 3:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[len(objects) // 2][2][1]), int(objects[len(objects) // 2][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]
        if adj in ['three_smallest', 'three_thinnest']:
            # 说明要在全部物体中找到最小的 这不简单？
            objects = []
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
                for name in size_classes[::-1]:
                    for detection in detections:
                        if detection[0] == name:
                            objects.append(detection)
                            break
                    if len(objects) >= 3:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[len(objects) // 2][2][1]), int(objects[len(objects) // 2][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]

        if adj == 'three_heaviest':
            # 说明要在全部物体中找到前3重的 这不简单？
            objects = []
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
                for name in weight_classes:
                    for detection in detections:
                        if detection[0] == name:
                            objects.append(detection)
                            break
                    if len(objects) >= 3:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[len(objects) // 2][2][1]), int(objects[len(objects) // 2][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]
        if adj == 'three_lightest':
            # 说明要在全部物体中找到最轻的 这不简单？
            objects = []
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
                for name in weight_classes[::-1]:
                    for detection in detections:
                        if detection[0] == name:
                            objects.append(detection)
                            break
                    if len(objects) >= 3:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[len(objects) // 2][2][1]), int(objects[len(objects) // 2][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]

    elif type_name != '' and adj != '':
        if adj in ['biggest', 'largest']:
            # 说明要在特定种类物体中找到最大的 这不简单？
            objects = []
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
                for name in size_classes:
                    for detection in detections:
                        if detection[0] == name and detection[0] in type_classes[type_name]:
                            objects.append(detection)
                            break
                    if len(objects) >= 1:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[0][2][1]), int(objects[0][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]
        if adj in ['smallest', 'thinnest']:
            # 说明要在全部物体中找到最小的 这不简单？
            objects = []
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
                for name in size_classes[::-1]:
                    for detection in detections:
                        if detection[0] == name and detection[0] in type_classes[type_name]:
                            objects.append(detection)
                            break
                    if len(objects) >= 3:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[0][2][1]), int(objects[0][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]

        if adj == 'heaviest':
            # 说明要在全部物体中找到最重的 这不简单？
            objects = []
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
                for name in weight_classes:
                    for detection in detections:
                        if detection[0] == name and detection[0] in type_classes[type_name]:
                            objects.append(detection)
                            break
                    if len(objects) >= 1:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[0][2][1]), int(objects[0][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]
        if adj == 'lightest':
            # 说明要在全部物体中找到最轻的 这不简单？
            objects = []
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
                for name in weight_classes[::-1]:
                    for detection in detections:
                        if detection[0] == name and detection[0] in type_classes[type_name]:
                            objects.append(detection)
                            break
                    if len(objects) >= 3:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[0][2][1]), int(objects[0][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]

        if adj in ['three_biggest', 'three_largest']:
            # 说明要在全部物体中找到前三大的 这不简单？
            objects = []
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
                for name in size_classes:
                    for detection in detections:
                        if detection[0] == name and detection[0] in type_classes[type_name]:
                            objects.append(detection)
                            break
                    if len(objects) >= 3:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[len(objects) // 2][2][1]), int(objects[len(objects) // 2][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]
        if adj in ['three_smallest', 'three_thinnest']:
            # 说明要在全部物体中找到最小的 这不简单？
            objects = []
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
                for name in size_classes[::-1]:
                    for detection in detections:
                        if detection[0] == name and detection[0] in type_classes[type_name]:
                            objects.append(detection)
                            break
                    if len(objects) >= 3:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[len(objects) // 2][2][1]), int(objects[len(objects) // 2][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]

        if adj == 'three_heaviest':
            # 说明要在全部物体中找到前3重的 这不简单？
            objects = []
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
                for name in weight_classes:
                    for detection in detections:
                        if detection[0] == name and detection[0] in type_classes[type_name]:
                            objects.append(detection)
                            break
                    if len(objects) >= 3:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[len(objects) // 2][2][1]), int(objects[len(objects) // 2][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]
        if adj == 'three_lightest':
            # 说明要在全部物体中找到最轻的 这不简单？
            objects = []
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
                for name in weight_classes[::-1]:
                    for detection in detections:
                        if detection[0] == name and detection[0] in type_classes[type_name]:
                            objects.append(detection)
                            break
                    if len(objects) >= 3:
                        break
                if len(objects) > 0:
                    # 说明找到了 就这？
                    name = ','.join([v[0] for v in objects])
                    print(colored("name:{}".format(name)))
                    coordinate = capture.transformed_depth_point_cloud[int(objects[len(objects) // 2][2][1]), int(objects[len(objects) // 2][2][0])]
                    if coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0:
                        # 说明坐标有效
                        coordinate = [v / 1000 for v in coordinate]
                    return name, coordinate
                if try_count >= 100:
                    return '', [-10, -10, -10]


def call_back(req):
    name, type_name, adj = req.name, req.type, req.adj
    res = xm_find_objectResponse()
    name, coordinate = deal_with_req(name, type_name, adj)
    res.name = name
    if coordinate[0] == -10:
        res.position.point.x, res.position.point.y, res.position.point.z = coordinate
    else:
        res.position.point.x = coordinate[2]
        res.position.point.y = -coordinate[0]
        res.position.point.z = -coordinate[1]
    res.position.header.frame_id = "kinect2_rgb_link"
    res.position.header.stamp = rospy.Time(0)
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

    size_classes = ['grape juice', 'coke', 'milk', 'orange juice', 'sprite', 'chocolate drink', 'sausage', 'orange', 'apple', 'potato chips', 'cracker', 'pringle', 'noodles']
    weight_classes = ['grape juice', 'coke', 'milk', 'orange juice', 'sprite', 'chocolate drink', 'sausage', 'orange', 'apple', 'potato chips', 'cracker', 'pringle', 'noodles']

    config_file1 = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/xm_vision.cfg'
    data_file1 = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/obj.data'
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
    network, class_names, class_colors = darknet.load_network(
        config_file2,
        data_file2,
        weights2,
        batch_size=batch_size2
    )
    width = darknet.network_width(network2)
    height = darknet.network_height(network2)
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
    rospy.init_node('findObjectNode')
    service = rospy.Service('findObject', xm_find_object, call_back)
    rospy.loginfo('GPSR Find Object\'s Vision Server Start!')
    rospy.spin()
