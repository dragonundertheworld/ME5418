from PIL import Image
import numpy as np
from dummy_gym import *

# # 示例 image_list，每个二维数组为灰度图
# image_list = [
#     np.random.randint(0, 50, (100, 100), dtype=np.uint8),
#     np.random.randint(0, 50, (100, 100), dtype=np.uint8),
#     np.random.randint(0, 50, (100, 100), dtype=np.uint8)
# ]

# env = DummyGym()
# image_list = []

# for i in range(4):
#     env.step(3)
#     processed_map = env.visit_count / np.max(env.visit_count) * 255
#     image_list.append(processed_map)

# # 将二维数组转换为图像对象
# pil_images = [Image.fromarray(image/image*255) for image in image_list]

# # 保存为 GIF，设置帧间隔
# output_path = "output.gif"
# pil_images[0].save(
#     output_path,
#     save_all=True,
#     append_images=pil_images[1:],  # 其余帧
#     duration=200,  # 每帧间隔（毫秒）
#     loop=0  # 循环次数，0 表示无限循环
# )

# print(f"GIF 保存成功：{output_path}")
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A[1:, 1:])