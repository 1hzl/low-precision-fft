@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d D:\cc\low-precision-fft
set DISTUTILS_USE_SDK=1
python build_ext.py build_ext --inplace
