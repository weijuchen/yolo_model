import cv2
import matplotlib.pyplot as plt
import easyocr
from matplotlib.font_manager import FontProperties
import xml.etree.ElementTree as ET

# 初始化 EasyOCR 讀取器，設置語言為繁體中文
reader = easyocr.Reader(['ch_tra'])

# 讀取影像
image_path = '2024-10-08-09.png'
image = cv2.imread(image_path)

# 進行文字識別
results = reader.readtext(image_path)

# 設置中文字體
font = FontProperties(fname='kaiu.ttf')

# 繪製結果
for (bbox, text, prob) in results:
    # 顯示邊界框
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    
    # 顯示文字
    plt.text(top_left[0], top_left[1] - 10, text, fontproperties=font, fontsize=12, color='green')

# 使用 matplotlib 顯示影像
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print(results)

# 建立 XML 結構
root = ET.Element("OCRResults")
for (bbox, text, prob) in results:
    entry = ET.SubElement(root, "Entry")
    ET.SubElement(entry, "Text").text = text
    ET.SubElement(entry, "Probability").text = str(prob)
    bbox_elem = ET.SubElement(entry, "BoundingBox")
    ET.SubElement(bbox_elem, "TopLeft").text = str(bbox[0])
    ET.SubElement(bbox_elem, "TopRight").text = str(bbox[1])
    ET.SubElement(bbox_elem, "BottomRight").text = str(bbox[2])
    ET.SubElement(bbox_elem, "BottomLeft").text = str(bbox[3])

# 將 XML 結構寫入檔案
tree = ET.ElementTree(root)
tree.write("ocr_results.xml", encoding='utf-8', xml_declaration=True)

print("OCR results saved to ocr_results.xml")