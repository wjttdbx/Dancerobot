import pyrealsense2 as rs
import cv2
import numpy as np

rsPipeline = rs.pipeline()
rsConfig = rs.config()
rsConfig.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
rsConfig.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
rsPipeline.start(rsConfig)

while True:
    frames = rsPipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    HSVimg = cv2.cvtColor(color_image , cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(HSVimg , (0 , 0 , 0) , (170 , 255 , 46))
    # print(mask)
    cv2.imshow("frame" , mask)
    depth_sum = 0
    sum = 0
    for i in range(640):
        for j in range(480):
            if mask[j][i] == 255 :
                if depth_image[j][i] <= 5000:
                    depth_sum += depth_image[j][i]
                    sum += 1
    if sum != 0:
        print(depth_sum / sum)

    k = cv2.waitKey(10)
    if k & 0xff == ord('q') or k == 27:# 按q或esc
        cv2.destoryAllWindows()
        break
