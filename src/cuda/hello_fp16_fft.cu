#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftXt.h>

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            std::exit(1);                                             \
        }                                                             \
    } while (0)

#define CHECK_CUFFT(call)                                             \
    do {                                                              \
        cufftResult err = (call);                                     \
        if (err != CUFFT_SUCCESS) {                                   \
            std::fprintf(stderr, "cuFFT error at %s:%d: %d\n",       \
                    __FILE__, __LINE__, (int)err);                    \
            std::exit(1);                                             \
        }                                                             \
    } while (0)

int main() {
    const long long n = 1024;
    const long long batch = 1;

    // Each complex element = 2 half values (real, imag)
    const size_t data_size = n * 2 * sizeof(__half);

    // Host input: build a signal with a known frequency component
    std::vector<__half> h_input(n * 2);
    for (long long i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(n);
        float val = std::sin(2.0f * M_PI * 8.0f * t);  // 8 Hz sine
        h_input[i * 2 + 0] = __float2half(val);        // real
        h_input[i * 2 + 1] = __float2half(0.0f);       // imag
    }

    // Device memory
    __half *d_input = nullptr;
    __half *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, data_size));
    CHECK_CUDA(cudaMalloc(&d_output, data_size));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), data_size,
                          cudaMemcpyHostToDevice));

    // Create cuFFT plan via Xt API
    cufftHandle plan;
    CHECK_CUFFT(cufftCreate(&plan));

    size_t workSize = 0;
    long long fft_n = n;

    CHECK_CUFFT(cufftXtMakePlanMany(
        plan,
        1,                    // rank
        &fft_n,               // n
        nullptr,              // inembed
        1,                    // istride
        n,                    // idist
        CUDA_C_16F,           // input type
        nullptr,              // onembed
        1,                    // ostride
        n,                    // odist
        CUDA_C_16F,           // output type
        batch,                // batch
        &workSize,            // workSize
        CUDA_C_16F            // execution type
    ));

    std::printf("cuFFT plan created: n=%lld, workSize=%zu bytes\n",
                n, workSize);

    // Set work area if needed
    void *d_work = nullptr;
    if (workSize > 0) {
        CHECK_CUDA(cudaMalloc(&d_work, workSize));
        CHECK_CUFFT(cufftXtSetWorkArea(plan, &d_work));
    }

    // Execute forward FFT
    CHECK_CUFFT(cufftXtExec(plan, d_input, d_output, CUFFT_FORWARD));

    // Copy result back to host
    std::vector<__half> h_output(n * 2);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, data_size,
                          cudaMemcpyDeviceToHost));

    // Verify non-zero output
    float max_mag = 0.0f;
    long long peak_bin = -1;
    for (long long i = 0; i < n; ++i) {
        float re = __half2float(h_output[i * 2 + 0]);
        float im = __half2float(h_output[i * 2 + 1]);
        float mag = std::sqrt(re * re + im * im);
        if (mag > max_mag) {
            max_mag = mag;
            peak_bin = i;
        }
    }

    std::printf("FFT complete. Peak magnitude: %f at bin %lld\n",
                max_mag, peak_bin);
    std::printf("Output at peak bin: %.6f + %.6fj\n",
                __half2float(h_output[peak_bin * 2 + 0]),
                __half2float(h_output[peak_bin * 2 + 1]));

    // Print first few bins
    std::printf("First 10 magnitude bins:\n");
    for (int i = 0; i < 10; ++i) {
        float re = __half2float(h_output[i * 2 + 0]);
        float im = __half2float(h_output[i * 2 + 1]);
        std::printf("  bin %2d: %.6f + %.6fj  (mag=%.6f)\n",
                    i, re, im, std::sqrt(re * re + im * im));
    }

    if (max_mag <= 0.0f) {
        std::fprintf(stderr, "ERROR: Output is all zeros!\n");
        std::exit(1);
    }

    std::printf("\nPASS: Non-zero FFT output verified.\n");

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    if (d_work) CHECK_CUDA(cudaFree(d_work));
    CHECK_CUFFT(cufftDestroy(plan));

    return 0;
}
