import cv2
import paddle
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from numpy import int64
import os
import time

# 初始化 PaddleOCR
ocr = PaddleOCR(det_model_dir="./output/det/infer", rec_model_dir="./output/CCPD/rec/infer", use_angle_cls=True,
                lang="ch")

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 获取视频的宽度、高度和帧率
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('车牌识别结果.avi', fourcc, fps, (frame_width, frame_height))

print("正在录制视频...按 'q' 键退出并保存视频")


# 检查并设置中文字体
def find_font_path():
    """查找系统中可用的中文字体"""
    possible_fonts = [
        "simhei.ttf",  # Windows
        "simfang.ttf",  # Windows
        "WenQuanYi Micro Hei",  # Linux
        "Heiti TC",  # macOS
    ]

    # 检查指定路径下的字体
    if os.path.exists('./fonts/simhei.ttf'):
        return './fonts/simhei.ttf'
    if os.path.exists('./fonts/simfang.ttf'):
        return './fonts/simfang.ttf'

    # 检查系统字体目录
    if os.name == 'nt':  # Windows
        font_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
        for font in possible_fonts[:2]:
            if os.path.exists(os.path.join(font_dir, font)):
                return os.path.join(font_dir, font)
    elif os.name == 'posix':  # Linux 或 macOS
        for font in possible_fonts[2:]:
            try:
                ImageFont.truetype(font, 12)
                return font
            except:
                continue

    print("警告: 未找到中文字体，将使用默认字体")
    return None


font_path = find_font_path()

# 记录上一次识别的车牌信息，用于去重显示
last_plate_info = None
# 上次更新时间
last_update_time = 0

while True:
    # 读取一帧
    ret, frame = cap.read()

    if not ret:
        print("无法获取帧")
        break

    # 转换BGR为RGB（Pillow使用RGB格式）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行检测和识别
    result = ocr.ocr(rgb_frame, cls=True)

    # 清空当前行（用于动态更新终端显示）
    print("\033[K", end="")

    # 在画面上绘制结果
    if result and len(result) > 0 and result[0]:
        # 提取检测框、文本和分数
        boxes = [line[0] for line in result[0]]
        txts = [line[1][0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]

        # 终端显示识别结果
        plate_info = []
        for i, (box, txt, score) in enumerate(zip(boxes, txts, scores)):
            plate_info.append(f"{txt} ({score:.2f})")

        # 只有当识别结果变化或超过2秒时才更新显示
        current_time = time.time()
        if plate_info != last_plate_info or current_time - last_update_time > 2:
            print(f"\r识别结果: {', '.join(plate_info)}", end="")
            last_plate_info = plate_info
            last_update_time = current_time

        # 使用Pillow绘制中文字符
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img)

        # 设置字体
        if font_path:
            try:
                font = ImageFont.truetype(font_path, 20)
            except:
                print(f"无法加载字体: {font_path}，使用默认字体")
                font = None
        else:
            font = None

        # 绘制检测框和文本
        for box, txt, score in zip(boxes, txts, scores):
            # 绘制检测框
            box = [(int(pt[0]), int(pt[1])) for pt in box]
            draw.polygon(box, outline=(0, 255, 0), width=2)

            # 计算文本位置（框的左上角）
            x, y = min([pt[0] for pt in box]), min([pt[1] for pt in box]) - 30

            # 绘制文本和置信度
            text = f"{txt} ({score:.2f})"
            draw.text((x, y), text, fill=(255, 0, 0), font=font)

        # 转换回OpenCV格式
        image_with_results = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 保存到视频
        output_video.write(image_with_results)

        # 显示结果
        cv2.imshow('车牌检测', image_with_results)
    else:
        # 如果没有检测到车牌，在终端显示提示
        print("\r未检测到车牌", end="")

        # 在画面上显示提示信息
        cv2.putText(frame, "未检测到车牌", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 保存到视频
        output_video.write(frame)

        # 显示结果
        cv2.imshow('车牌检测', frame)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print()  # 换行，使退出信息更美观
        break

# 释放资源
cap.release()
output_video.release()
cv2.destroyAllWindows()

print("视频已保存为: 车牌识别结果.avi")