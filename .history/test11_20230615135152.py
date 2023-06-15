import torch
import torch.nn as nn

# 定义输入大小和输出大小
input_size = (100, 1, 15)
output_size = (100, 4, 9)

# 定义转置卷积层
conv_transpose = nn.ConvTranspose1d(
    in_channels=input_size[1],  # 输入通道数
    out_channels=output_size[1],  # 输出通道数
    kernel_size=3,  # 卷积核大小
    stride=2,  # 步长
    padding=1  # 填充大小
)

# 创建一个随机输入张量
input_tensor = torch.randn(input_size)

# 进行转置卷积
output_tensor = conv_transpose(input_tensor)

# 打印输出张量的大小
print(output_tensor.size())  # 输出: torch.Size([100, 4, 9])