@echo off
REM Build BFP FFT CUDA kernel from VS Developer Command Prompt
REM Run from project root: build_bfp.bat

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set NVCC=%CUDA_PATH%\bin\nvcc.exe

if not exist build mkdir build

%NVCC% -arch=sm_120 -O3 -o build\bfp_fft.exe src\cuda\bfp_fft.cu
if %ERRORLEVEL% EQU 0 (
    echo [OK] build\bfp_fft.exe
    echo Run: build\bfp_fft.exe
) else (
    echo [FAIL] Compilation failed
)
