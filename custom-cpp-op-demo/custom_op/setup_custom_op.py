from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='custom_setup_ops',
    ext_modules=CUDAExtension(
        sources=['custom_relu_cpu.cc',
                 'custom_relu_cuda.cc', 
                 'custom_relu_cuda.cu']
    )
)

