import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def add_text_to_image(
    input_path,
    output_path,
    text,
    position=(50, 50),
    font_path="/workspace/yh/project/CGHReview/utils/Times New Roman Bold Italic.ttf",
    font_size=48,
    color=(255, 0, 0)
):
    """
    在图片上添加指定字体文字（支持 Times New Roman Italic）

    参数：
        input_path: 输入图片路径
        output_path: 输出图片路径
        text: 添加的文字
        position: 文字左上角坐标 (x, y)
        font_path: 字体文件路径
        font_size: 字体大小
        color: RGB颜色，例如红色=(255,0,0)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入图片不存在: {input_path}")

    if not os.path.exists(font_path):
        raise FileNotFoundError(f"字体文件不存在: {font_path}")

    # 用 OpenCV 读图
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法读取图片: {input_path}")

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 转成 PIL 图像
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)

    # 写字
    draw.text(position, text, font=font, fill=color)

    # PIL -> OpenCV
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(output_path, result)
    print(f"已保存到: {output_path}")


if __name__ == "__main__":
    input_image = "/workspace/yh/project/CGHReview/result/Mesh3_n/50k_4/recon_rgb_crop1_ROI/recon_rgb_z_5.040000.png"
    output_image = "/workspace/yh/project/CGHReview/result/Mesh3_n/50k_4/recon_rgb_crop1_ROI/recon_rgb_z_5.040000_text.png"
    text = "focus"

    add_text_to_image(
        input_path=input_image,
        output_path=output_image,
        text=text,
        position=(2400, 3800),
        font_path="/workspace/yh/project/CGHReview/utils/Times New Roman Bold Italic.ttf",
        font_size=200,
        color=(255, 255, 255)   # RGB 白色
    )

    '''
    80, 50 

    3600, 50

    2400, 3800 
    
    '''