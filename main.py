from ultralytics import YOLO
import cv2

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 訓練參數
data = "D:\D_Download\AI保險理賠助手\dataset.yaml"  # 資料集配置文件
model = YOLO('yolov8s.pt')  # 預訓練權重文件

# 開始訓練
model.train(data=data, epochs=10)

# 驗證參數
data = 'D:\D_Download\AI保險理賠助手\dataset.yaml'  # 資料集配置文件

# 開始驗證
model.val(data=data)

# 預測參數
img_path = (
    "D:\D_Download\AI保險理賠助手\LINE_ALBUM_診斷證明書_241016_10.jpg"  # 要檢測的影像
)

# 開始檢測
results = model(img_path)

# 顯示結果
img = cv2.imread(img_path)
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 保存影像到文件
output_path = "result.jpg"
cv2.imwrite(output_path, img)
print(f"Result saved to {output_path}")
# cv2.imshow('Result', img)
# cv2.waitKey(0)


