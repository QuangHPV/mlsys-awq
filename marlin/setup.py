import os
import shutil
import subprocess
import sys


def _log(msg):
    # printed to stderr so it surfaces in uv's build log even on success
    print(f"[marlin/setup.py] {msg}", file=sys.stderr)


def _resolve_cuda_home():
    """Locate a CUDA toolkit (nvcc + headers) for the build, logging each attempt.
    Order: explicit env -> nvcc on PATH -> /usr/local/cuda -> pip nvidia-cuda-nvcc-cu12.
    Must run BEFORE importing torch.utils.cpp_extension, which caches CUDA_HOME from
    the environment at import time.
    """
    env = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env:
        _log(f"CUDA_HOME from env: {env}")
        return env

    nvcc = shutil.which("nvcc")
    if nvcc:
        home = os.path.dirname(os.path.dirname(nvcc))  # .../bin/nvcc -> toolkit root
        _log(f"nvcc found on PATH ({nvcc}) -> CUDA_HOME={home}")
        return home

    if os.path.exists("/usr/local/cuda/bin/nvcc"):
        _log("system toolkit found at /usr/local/cuda")
        return "/usr/local/cuda"

    try:  # pip wheel: nvcc only, no full headers/libs (may fail the header check below)
        import nvidia.cuda_nvcc  # nvidia.* are PEP 420 namespaces -> use __path__, not __file__
        home = list(nvidia.cuda_nvcc.__path__)[0]
        _log(f"pip nvidia-cuda-nvcc-cu12 found -> CUDA_HOME={home}")
        return home
    except (ImportError, IndexError):
        _log("pip nvidia-cuda-nvcc-cu12 not importable")

    return None


def _diagnose(cuda_home):
    """Report whether this CUDA_HOME can actually build: nvcc runs, headers present."""
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    _log(f"nvcc present: {os.path.exists(nvcc)} ({nvcc})")
    if os.path.exists(nvcc):
        try:
            out = subprocess.check_output([nvcc, "--version"], text=True, stderr=subprocess.STDOUT)
            _log("nvcc --version: " + out.strip().splitlines()[-1])
        except Exception as e:
            _log(f"nvcc --version failed: {e}")
    # the common pip-nvcc failure: compiler present but CUDA headers live elsewhere
    inc = os.path.join(cuda_home, "include")
    missing = [h for h in ("cuda.h", "cuda_runtime.h", "cuda_fp16.h")
               if not os.path.exists(os.path.join(inc, h))]
    if missing:
        _log(f"WARNING: headers missing in {inc}: {missing} -> compile will likely fail; "
             "use a full CUDA toolkit (conda install -c nvidia cuda-toolkit, or set CUDA_HOME).")
    else:
        _log(f"CUDA headers OK in {inc}")


_cuda_home = _resolve_cuda_home()
if _cuda_home:
    os.environ["CUDA_HOME"] = _cuda_home
    _diagnose(_cuda_home)
else:
    _log("ERROR: no CUDA toolkit found (need nvcc + headers). Set CUDA_HOME to a CUDA "
         "toolkit root, or `conda install -c nvidia cuda-toolkit=12.1`.")

# Import AFTER CUDA_HOME is set: cpp_extension reads/caches it at import time.
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='marlin',
    version='0.1.1',
    author='Elias Frantar',
    author_email='elias.frantar@ist.ac.at',
    description='Highly optimized FP16xINT4 CUDA matmul kernel.',
    install_requires=['numpy', 'torch'],
    packages=['marlin'],
    package_dir={'marlin': '.'},  # flat layout: setup.py dir is itself the marlin package
    ext_modules=[cpp_extension.CUDAExtension(
        'marlin.marlin_cuda', ['marlin_cuda.cpp', 'marlin_cuda_kernel.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)