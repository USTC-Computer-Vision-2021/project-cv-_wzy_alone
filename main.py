# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

# # paryⅠ 检测标记
# aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
# aruco_marker_1 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=1)
# aruco_marker_2 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=2)
# aruco_marker_3 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=3)
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(aruco_marker_1)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(1, 3, 2)
# plt.imshow(aruco_marker_2)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(1, 3, 3)
# plt.imshow(aruco_marker_3)
# plt.xticks([])
# plt.yticks([])
# plt.show()
# # 创建字典对象
# aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
# # 创建参数对象
# parameters = cv2.aruco.DetectorParameters_create()
# # 创建视频捕获对象
# capture = cv2.VideoCapture(0)
# while True:
#     # 捕获视频帧
#     ret, frame = capture.read()
#     # 转化为灰度图像
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # 检测图像中标记
#     corners, ids, rejected_corners = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)
#     # 绘制检测标记
#     frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))
#     # 绘制被拒绝标记
#     frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejected_corners, borderColor=(0, 0, 255))
#     # 展示结果
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# # 销毁窗口
# capture.release()
# cv2.destroyAllWindows()


# # partⅡ 相机校准
# # 创建板
# dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
# board = cv2.aruco.CharucoBoard_create(3, 3, .025, .0125, dictionary)
# img = board.draw((200 * 3, 200 * 3))
# cv2.imshow("board", img)
# cv2.waitKey(0)
# # 创建字典对象
# dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
# # 创建视频捕获对象
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# all_corners = []
# all_ids = []
# counter = 0
# for i in range(300):
#     # 读取帧
#     ret, frame = cap.read()
#     # 转化为灰度图像
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # 标记检测
#     res = cv2.aruco.detectMarkers(gray, dictionary)
#     if len(res[0]) > 0:
#         res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
#         if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and counter % 3 == 0:
#             all_corners.append(res2[1])
#             all_ids.append(res2[2])
#         # 绘制探测标记
#         cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     counter += 1
# try:
#     # 相机校准
#     cal = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, gray.shape, None, None)
# except:
#     cap.release()
#     print("Calibration could not be done ...")
# # 获取校准结果
# retval, cameraMatrix, distCoeffs, rvecs, tvecs = cal
# f = open('calibration.pckl', 'wb')
# pickle.dump((cameraMatrix, distCoeffs), f)
# f.close()
# print("over")


# # partⅢ 相机姿态估计
# # 加载相机校准数据
# with open('calibration.pckl', 'rb') as f:
#     cameraMatrix, distCoeffs = pickle.load(f)
# # 创建字典对象
# aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
# # 创建参数对象
# parameters = cv2.aruco.DetectorParameters_create()
# # 创建视频捕获对象
# capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# while True:
#     # 捕获视频帧
#     ret, frame = capture.read()
#     # 转换为灰度图像
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # 探测标记
#     corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)
#     # 绘制标记
#     frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))
#     # 绘制被拒绝候选标记
#     frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejectedImgPoints, borderColor=(0, 0, 255))
#     if ids is not None:
#         # rvecs, tvecs分别是角点中每个标记的旋转和平移向量
#         rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)
#         # 绘制系统轴
#         for rvec, tvec in zip(rvecs, tvecs):
#             cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 1)
#     # 绘制结果帧
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# partⅣ 增强现实
def draw_augmented_overlay(pts_1, overlay_image, image):
    """ 增强现实 """
    # 定义要绘制的叠加图像的正方形
    pts_2 = np.float32([[0, 0], [overlay_image.shape[1], 0], [overlay_image.shape[1], overlay_image.shape[0]],
                        [0, overlay_image.shape[0]]])
    # 绘制边框以查看图像边框
    cv2.rectangle(overlay_image, (0, 0), (overlay_image.shape[1], overlay_image.shape[0]), (255, 255, 0), 10)
    # 创建转换矩阵
    M = cv2.getPerspectiveTransform(pts_2, pts_1)
    # 使用变换矩阵M变换融合图像
    dst_image = cv2.warpPerspective(overlay_image, M, (image.shape[1], image.shape[0]))
    # 创建掩码
    dst_image_gray = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(dst_image_gray, 0, 255, cv2.THRESH_BINARY_INV)
    # 使用计算出的掩码计算按位与
    image_masked = cv2.bitwise_and(image, image, mask=mask)
    # 两个图像进行加和创建结果图像
    result = cv2.add(dst_image, image_masked)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    return result


OVERLAY_SIZE_PER = 1
image = cv2.imread('example.png')
# 加载相机校准数据
with open('calibration.pckl', 'rb') as f:
    cameraMatrix, distCoeffs = pickle.load(f)
# 创建字典对象
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
# 创建参数对象
parameters = cv2.aruco.DetectorParameters_create()
# 创建视频捕获对象
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    # 捕获视频帧
    ret, frame = capture.read()
    # 转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 探测标记
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)
    # 绘制标记
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))
    if ids is not None:
        # rvecs, tvecs分别是角点中每个标记的旋转和平移向量
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)
        for rvec, tvec in zip(rvecs, tvecs):
            # 定义要覆盖图像的点
            desired_points = np.float32(
                [[-1 / 2, 1 / 2, 0], [1 / 2, 1 / 2, 0], [1 / 2, -1 / 2, 0], [-1 / 2, -1 / 2, 0]]) * OVERLAY_SIZE_PER
            # 投影点
            projected_desired_points, jac = cv2.projectPoints(desired_points, rvecs, tvecs, cameraMatrix, distCoeffs)
            # 绘制投影点
            draw_augmented_overlay(projected_desired_points, image, frame)
    # 绘制结果帧
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
desired_points = np.float32([[-1 / 2, 1 / 2, 0], [1 / 2, 1 / 2, 0], [1 / 2, -1 / 2, 0], [-1 / 2, -1 / 2, 0]]) * OVERLAY_SIZE_PER
projected_desired_points, jac = cv2.projectPoints(desired_points, rvecs, tvecs, cameraMatrix, distCoeffs)


