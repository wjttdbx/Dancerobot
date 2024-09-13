import cv2
import pyrealsense2 as rs
import numpy as np

rsPipeline = rs.pipeline()
rsConfig = rs.config()
rsConfig.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
rsConfig.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
rsPipeline.start(rsConfig)

for i in range(100):
    frames = rsPipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())



    HSVimg = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)

    img = cv2.circle(HSVimg , (320 , 240) , 5 , (255 , 0 , 0))

    print(HSVimg[320 , 240])

    cv2.imshow("frame" , img)

    k = cv2.waitKey(10)
    if k & 0xff == ord('q') or k == 27:
        break