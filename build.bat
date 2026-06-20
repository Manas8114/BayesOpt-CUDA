@echo off
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "CUDA_PATH=%CUDA_HOME%"
set DISTUTILS_USE_SDK=1
set USE_NINJA=0
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
python setup.py build_ext --inplace
