import paddle
from paddle.autograd import PyLayer
import numpy as np


# Relu: Y = max(0, X)
# 通过创建`PyLayer`子类的方式实现动态图自定义Python算子
class custom_relu(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, x): 
        np_result = np.maximum(0, x.numpy())
        y = paddle.to_tensor(np_result, stop_gradient=False)
        # ctx 为PyLayerContext对象，可以把x从forward传递到backward
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        # ctx 为PyLayerContext对象，saved_tensor获取在forward时暂存的x
        x, = ctx.saved_tensor()
        np_tmp = x.numpy()
        np_tmp[np_tmp > 0] = 1
        np_tmp[np_tmp <= 0] = 0
        grad = dy * paddle.to_tensor(np_tmp)
        return grad
    