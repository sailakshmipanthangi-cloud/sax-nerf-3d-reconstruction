from torch.utils.cpp_extension import load
import os

_src_path = os.path.dirname(__file__)

_backend = load(
    name='hashencoder_backend',
    sources=[
        os.path.join(_src_path, 'src/bindings.cpp'),
        os.path.join(_src_path, 'src/hashencoder.cu'),
    ],
    extra_cuda_cflags=['-allow-unsupported-compiler'],
    verbose=True
)