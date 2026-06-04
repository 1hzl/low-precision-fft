@echo off
REM Build FP8 verification program from VS Developer Command Prompt
REM Run from project root: build_fp8.bat

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set NVCC=%CUDA_PATH%\bin\nvcc.exe

if not exist build mkdir build

%NVCC% -arch=sm_120 -O3 -o build\fp8_verification.exe src\cuda\fp8_verification.cu
if %ERRORLEVEL% EQU 0 (
    echo [OK] build\fp8_verification.exe
    echo Run: build\fp8_verification.exe
) else (
    echo [FAIL] Compilation failed
)
