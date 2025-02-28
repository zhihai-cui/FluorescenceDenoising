from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# # 创建一个 NumPy 数组，表示灰度图像
# image_array = np.array([[0, 1], [128, 255]])
# # 使用 Image.fromarray() 将 NumPy 数组转换为 PIL 图像对象
# image = Image.fromarray(image_array)
# # 显示图像
# plt.imshow(image_array, cmap='gray')
# plt.show()


# 打开图片
img = Image.open(r'E:\cuizhihai\DenoisingNe2Ne\test\fm_img\mito2_gt.png')

print(img.size)
print(img.mode)

# 获取并打印图片的通道数
print(len(img.split())) 





r, g, b = img.split()
channels = [r, g, b]



# img = np.squeeze(img)
# print(img.shape)

for channel in channels:
    channel = np.array(channel)
    print(channel.shape)
    
    

# img = np.array(img)
# print(img.shape)

# img = np.squeeze(img)
# print(img.shape)

# # 获取并打印图片的通道数
# print(len(img.split())) # 1


