import cv2
import numpy as np
import glob

# 1. 读取所有静态缺陷图片（假设放在 images 文件夹下）
img_paths = sorted(glob.glob("D:/YAN/竞赛/2026能源装备创新设计大赛/Data set/Welding defect.v3-binov-defect-dataset.yolov8/test/images/*.jpg"))
frames_list = []

# 2. 模拟相机平移：对每张图进行动态裁剪过渡
for path in img_paths:
    # 将 img = cv2.imread(path) 替换为以下这行：
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

    # 然后为了防止解码失败还是报错，务必加上安全判断：
    if img is None:
        print(f"⚠️ 警告: 无法读取图片，已跳过 -> {path}")
        continue  # 跳过这张坏图，继续处理下一张
    img = cv2.resize(img, (640, 480))
    h, w, c = img.shape

    # 通过在单张图上做微小的像素位移，模拟相机抖动或微调扫描
    for offset in range(0, 50, 2):
        # 创建一个平移矩阵
        M = np.float32([[1, 0, offset], [0, 1, 0]])
        shifted_frame = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        frames_list.append(shifted_frame)

# 3. 将所有生成的时序帧保存为标准 H.264 编码视频
fourcc = cv2.VideoWriter_fourcc(*'X264')  # 兼容浏览器播放
out = cv2.VideoWriter('weld_test_stream.mp4', fourcc, 25.0, (640, 480))

for frame in frames_list:
    out.write(frame)
out.release()
print("🎉 成功生成用于跨帧追踪的动态焊缝视频流！")