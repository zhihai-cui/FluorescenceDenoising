from PIL import Image

# 打开图片
img = Image.open('E:/cuizhihai/fmdd/test_mix/avg2/Confocal_BPAE_B_1.png')

# 获取并打印图片的通道数
print(len(img.split())) # 1


