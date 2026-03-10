from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hashencoder_backend',
    ext_modules=[
        CUDAExtension(
            name='hashencoder_backend',
            sources=['bindings.cpp', 'hashencoder.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })