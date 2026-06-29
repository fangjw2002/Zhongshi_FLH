import cv2
import numpy as np
from ultralytics import YOLO


# 导入你们之前写的骨架算法类
# from your_script import GeometricQuantifier

class WeldEngine:
    def __init__(self, weights_path='best.pt'):
        # 1. 加载模型
        self.model = YOLO(weights_path)
        self.mm_per_pixel = 0.12  # 比例尺

    def process_frame(self, frame):
        """处理单帧图像并返回标注图和数据"""
        # 2. 推理
        results = self.model.predict(frame, conf=0.4, verbose=False)
        res = results[0]

        # 3. 提取数据
        annotated_frame = res.plot()  # 获取带有渲染框的画面

        defects_data = []
        if res.masks is not None:
            for i in range(len(res.boxes)):
                mask = res.masks.xy[i]
                cls = self.model.names[int(res.boxes.cls[i])]
                conf = float(res.boxes.conf[i])

                # 计算像素面积
                pixel_area = cv2.contourArea(mask.astype(np.int32))
                # 换算物理面积
                mm2_area = pixel_area * (self.mm_per_pixel ** 2)

                defects_data.append({
                    "type": cls,
                    "conf": conf,
                    "area": mm2_area
                })

        return annotated_frame, defects_data