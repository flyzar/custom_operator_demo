import paddle
import paddle.nn.functional as F

# 方法1: 使用setuptools编译安装后导入 
# --------------------------------------------------------------------
# from custom_setup_ops import custom_relu, custom_relu_cpu
# --------------------------------------------------------------------

# 方法2: 使用即时编译（JIT Compile）导入 
# --------------------------------------------------------------------
from paddle.utils.cpp_extension import load
custom_ops = load(
    name="custom_jit_ops",
    sources=["custom_op/custom_relu_cuda.cc", # 建议使用绝对路径
             "custom_op/custom_relu_cuda.cu", 
             "custom_op/custom_relu_cpu.cc"])

custom_relu = custom_ops.custom_relu
custom_relu_cpu = custom_ops.custom_relu_cpu


# 引入自定义C++算子模型代码
class LeNet(paddle.nn.Layer):
    def __init__(self, device = 'GPU'):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)
        self.relu = custom_relu if device == 'GPU' else custom_relu_cpu # 使用自定义算子

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x) 
        x = self.max_pool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

# 原始模型代码
# class LeNet(paddle.nn.Layer):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
#         self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
#         self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
#         self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
#         self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
#         self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
#         self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.max_pool1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = self.max_pool2(x)
#         x = paddle.flatten(x, start_axis=1,stop_axis=-1)
#         x = self.linear1(x)
#         x = F.relu(x)
#         x = self.linear2(x)
#         x = F.relu(x)
#         x = self.linear3(x)
#         return x