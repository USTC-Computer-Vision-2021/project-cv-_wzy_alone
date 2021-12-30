[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6630551&assignment_repo_type=AssignmentRepo)
# 基于增强现实技术的基础使用

成员及分工
- 王泽雨 PB18061235
  - 单人队伍


## 问题描述

  增强现实 (Augmented Reality, AR) 是目前最热门的应用研究之一。增强现实的概念可以定义为将虚拟信息与真实世界巧妙融合的技术，其中真实世界的视图通过叠加的计算机生成的虚拟元素(例如，图像、视频或 3D 模型等)得到增强。为了叠加和融合数字信息(增强现实)，可以使用不同类型的技术，主要包括基于位置和基于识别的方法。选择做这个是对视频马赛克打码感到兴趣，想要复现类似的效果，试图用别的内容替换掉“马赛克”的内容，比如用表情等。

## 原理分析

基于识别的增强现实使用图像处理技术来推导出用户正在查看的位置。从图像中获取相机姿态需要找到环境中已知点与其对应的相机投影之间的对应关系。查找这种对应关系的方法可以分为两类：

- 基于标记的姿态估计：这种方法依赖于使用平面标记(在增强现实领域多使用基于方形标记的方法)计算相机姿态。使用方形标记的一个主要缺点是相机姿态的计算依赖于对标记四个角的准确确定。在有遮挡的情况下，标记可能非常困难，但目前已有一些基于标记检测的方法也可以很好地处理遮挡，例如 ArUco。

- 基于无标记的姿态估计：当场景不能使用标记来获得姿态估计时，可以使用图像中自然存在的对象进行姿态估计。一旦计算了一组 n 个 2D 点及其相应的 3D 坐标，就可以通过解决透视 n 点 (Perspective-n-Point, PnP) 问题来估计相机的姿态。由于这些方法依赖于点匹配技术，但输入数据中大多包含异常值，因此在姿态估计过程中需要使用针对异常值的鲁棒技术(例如，RANSAC)。

本次实验基于标记的增强现实的工作原理，使用ArUco算法基于标记增强现实。ArUco 会自动检测标记并纠正可能的错误。此外，ArUco 提出了通过将多个标记与遮挡掩码组合来解决遮挡问题的方法，遮挡掩码是通过颜色分割计算的。使用标记的主要好处是可以在图像中有效且稳健基于标记执行姿势估计，其可以准确地导出标记的四个角，从先前计算的标记的四个角中获得相机姿态。



## 代码实现

### 第一步进行相机校准

####  paryⅠ检测标记

```python
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
aruco_marker_1 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=1)
aruco_marker_2 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=2)
aruco_marker_3 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=3)
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(aruco_marker_1)
plt.xticks([])
plt.yticks([])
plt.subplot(1, 3, 2)
plt.imshow(aruco_marker_2)
plt.xticks([])
plt.yticks([])
plt.subplot(1, 3, 3)
plt.imshow(aruco_marker_3)
plt.xticks([])
plt.yticks([])
plt.show()

创建的标记进行可视化，结果如下

![image](https://github.com/USTC-Computer-Vision-2021/project-cv-_wzy_alone/blob/main/images/1.jpg)

使用cv2.aruco.detectMarkers()函数检测标记，然后使用 cv2.aruco.drawDetectedMarkers() 函数绘制检测到的标记和拒绝的候选标记。

# 创建字典对象
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
# 创建参数对象
parameters = cv2.aruco.DetectorParameters_create()
# 创建视频捕获对象
capture = cv2.VideoCapture(0)
while True:
    # 捕获视频帧
    ret, frame = capture.read()
    # 转化为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检测图像中标记
    corners, ids, rejected_corners = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)
    # 绘制检测标记
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))
    # 绘制被拒绝标记
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejected_corners, borderColor=(0, 0, 255))
    # 展示结果
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 销毁窗口
capture.release()
cv2.destroyAllWindows()
```

执行程序后，可以看到检测到的标记用绿色边框绘制，而被拒绝的候选标记用红色边框绘制。

![image](https://github.com/USTC-Computer-Vision-2021/project-cv-_wzy_alone/blob/main/images/2.jpg)

#### partⅡ相机校准

在使用检测到的标记获得相机姿态之前，需要知道相机的标定参数，aruco 提供了执行此任务所需的校准程序，校准程序仅需执行一次，因为程序执行过程中并未修改相机光学元件。校准过程中使用的主要函数是 cv2.aruco.calibrateCameraCharuco()，其使用从板上提取的多个视图中的一组角来校准相机。校准过程完成后，此函数返回相机矩阵(一个 3 x 3 浮点相机矩阵)和一个包含失真系数的向量，3 x 3 矩阵对焦距和相机中心坐标(也称为内在参数)进行编码，而失真系数对相机产生的失真进行建模。

这是以下过程中创建的标记板

![image](https://github.com/USTC-Computer-Vision-2021/project-cv-_wzy_alone/blob/main/images/3.jpg)

校准过程完成后，使用pickle将相机矩阵和失真系数保存到磁盘。之后就可以执行相机姿态估计过程。

![image](https://github.com/USTC-Computer-Vision-2021/project-cv-_wzy_alone/blob/main/images/4.jpg)

```python
# 创建板
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
board = cv2.aruco.CharucoBoard_create(3, 3, .025, .0125, dictionary)
img = board.draw((200 * 3, 200 * 3))
cv2.imshow("board", img)
cv2.waitKey(0)
# 创建字典对象
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
# 创建视频捕获对象
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
all_corners = []
all_ids = []
counter = 0
for i in range(300):
    # 读取帧
    ret, frame = cap.read()
    # 转化为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 标记检测
    res = cv2.aruco.detectMarkers(gray, dictionary)
    if len(res[0]) > 0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and counter % 3 == 0:
            all_corners.append(res2[1])
            all_ids.append(res2[2])
        # 绘制探测标记
        cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    counter += 1
try:
    # 相机校准
    cal = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, gray.shape, None, None)
except:
    cap.release()
    print("Calibration could not be done ...")
# 获取校准结果
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cal
f = open('calibration.pckl', 'wb')
pickle.dump((cameraMatrix, distCoeffs), f)
f.close()
print("over")
```

#### partⅢ相机姿态估计

为了估计相机姿态，需要使用 cv2.aruco.estimatePoseSingleMarkers() 函数，用于估计单个标记的姿态，此外还需要使用cv2.aruco.drawAxis() 函数，用于为每个检测到的标记绘制系统轴。

```
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
    # 绘制被拒绝候选标记
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejectedImgPoints, borderColor=(0, 0, 255))
    if ids is not None:
        # rvecs, tvecs分别是角点中每个标记的旋转和平移向量
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)
        # 绘制系统轴
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 1)
    # 绘制结果帧
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```
结果如下

![image](https://github.com/USTC-Computer-Vision-2021/project-cv-_wzy_alone/blob/main/images/5.jpg)

#### partⅣ增强现实

draw_augmented_overlay() 函数首先定义叠加图像的正方形。然后计算变换矩阵，用于变换叠加图像得到 dst_image 图像；接下来，创建掩码并使用之前创建的掩码按位运算以获得 image_masked 图像；最后将 dst_image 和image_masked 相加，得到结果图像，并返回。
程序运行结果如下图所示：

![image](https://github.com/USTC-Computer-Vision-2021/project-cv-_wzy_alone/blob/main/images/6.png)

```python
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
```

## 效果展示

在这儿可以展示自己基于素材实现的效果，可以贴图，如果是视频，建议转成 Gif 插入，例如：

![AR 效果展示](demo/ar.gif)

如果自己实现了好玩儿的 feature，比如有意思的交互式编辑等，可以想办法展示和凸显出来。

## 工程结构

```text
.
├── code
│   ├── run.py
│   └── utils.py
├── input
│   ├── bar.png
│   └── foo.png
└── output
    └── result.png
```

## 运行说明

在这里，建议写明依赖环境和库的具体版本号，如果是 python 可以建一个 requirements.txt，例如：

```
opencv-python==3.4
Flask==0.11.1
```

运行说明尽量列举清晰，例如：
```
pip install opencv-python
python run.py --src_path xxx.png --dst_path yyy.png
npm run make-es5 --silent
```
