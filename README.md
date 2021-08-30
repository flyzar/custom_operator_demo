# custom_operator_demo
 PaddlePaddle自定义算子Demo

---

*C++自定义算子demo*

```
|-- custom-cpp-op-demo
    |-- custom-cpp 
        |-- custom_relu_cpu.cc  // CPU自定义算子
        |-- custom_relu_cuda.cc // CUDA自定义算子
        |-- custom_relu_cuda.cu // CUDA自定义算子
        |-- install.sh          // 执行脚本
        |-- setup_custom_op.py  // 编译安装配置脚本
    |-- model.py    // 网络模型定义
    |-- train.py    // 训练代码
    |-- run.sh      // 训练执行脚本
```


---

*Python自定义算子demo*

```
|-- custom-python-op-demo
    |-- custom_py_relu.py  // Python自定义relu算子
    |-- model.p            // 网络模型定义
    |-- train.py           // 训练代码
    |-- run.sh             // 训练执行脚本
```

运行代码 `python train.py`

