# 导入包和版本查询
import PIL
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import torchvision

def print_version_info():
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0),torch.cuda.get_device_properties(0))
    print(torch.cuda_path,torch.cuda_version)

# 可复现性

# 显卡设置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# 如果需要多张显卡
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# 清除显存
# torch.cuda.empty_cache()

#########张量处理 PyTorch有9种CPU张量类型和9种GPU张量类型。###############
tensor = torch.randn(3,4,5)
# print(tensor.type())  # 数据类型  torch.FloatTensor
# print(tensor.to(device).type()) # torch.cuda.FloatTensor
# print(tensor.cuda().type()) # torch.cuda.FloatTensor
# print(tensor.cpu().type()) # torch.FloatTensor
# print(tensor.size())  # 张量的shape，是个元组
# print(tensor.dim())   # 维度的数量

# tensor 与 np.ndarray 转换
ndarray = tensor.cpu().numpy()
tensor = torch.from_numpy(ndarray).float()


# Torch.tensor与PIL.Image转换
# pytorch中的张量默认采用[N, C, H, W]的顺序，并且数据范围在[0,1]，需要进行转置和规范化
# torch.Tensor -> PIL.Image
# image = PIL.Image.fromarray(torch.clamp(tensor*255, min=0, max=255).byte().permute(1,2,0).cpu().numpy())
# image = torchvision.transforms.functional.to_pil_image(tensor)  # Equivalently way
# image.show()

# PIL.Image -> torch.Tensor
# path = r'./figure.jpg'
# tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))).permute(2,0,1).float() / 255
# tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path)) # Equivalently way


# np.ndarray 与 PIL.Image 转换
# 成功的要求 ： 1. dtype是uint8；2. shape是(H, W, C)类型。
# a = np.random.rand(224,224,3)
# # a = np.clip(a,0,1) # 出错时，可以添加这个  将numpy数组约束在[0, 1]范围内
# image = PIL.Image.fromarray((a * 255).astype(np.uint8))
# image.show()

# 这里如果继续用上面的ndarry，需要进行维度转换 (C, H, W)->(H, W, C)
# 这里注意 transpose 和 reshape 是不一样的
# image = PIL.Image.fromarray((ndarray.transpose(1,2,0) * 255).astype(np.uint8))
# image.show()
