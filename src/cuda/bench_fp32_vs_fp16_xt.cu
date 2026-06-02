// Benchmark: FP32 vs FP16 cuFFT using ONLY Xt API for both
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftXt.h>

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t e = (call);                                       \
        if (e != cudaSuccess) {                                       \
            std::fprintf(stderr, "CUDA %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(e));      \
            std::exit(1);                                             \
        }                                                             \
    } while (0)

#define CHECK_CUFFT(call)                                             \
    do {                                                              \
        cufftResult r = (call);                                       \
        if (r != CUFFT_SUCCESS) {                                     \
            std::fprintf(stderr, "FFT %s:%d: %d\n",                  \
                    __FILE__, __LINE__, (int)r);                      \
            std::exit(1);                                             \
        }                                                             \
    } while (0)

// CPU fallback
std::vector<float> dft_cpu(const std::vector<float>& sig, long long n) {
    std::vector<float> out(n * 2, 0.0f);
    for (long long k = 0; k < n; ++k) {
        float re = 0, im = 0;
        for (long long i = 0; i < n; ++i) {
            float angle = -2.0f * M_PI * k * i / n;
            float s = sig[i * 2 + 0];
            float c = sig[i * 2 + 1];
            re += s * std::cos(angle) - c * std::sin(angle);
            im += s * std::sin(angle) + c * std::cos(angle);
        }
        out[k * 2 + 0] = re;
        out[k * 2 + 1] = im;
    }
    return out;
}

int main() {
    std::printf("FP32 vs FP16 cuFFT Benchmark (Xt API only)\n");
    std::printf("GPU: RTX 5070 Ti Laptop, CUDA 13.3\n\n");
    std::fflush(stdout);

    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    long long sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384,
                         32768, 65536, 131072, 262144, 524288, 1048576};
    int nsizes = 13;

    std::ofstream csv("data/bench-fp32-vs-fp16-xt.csv");
    csv << "n,fp32_us,fp16_us,speedup,max_rel_error,mean_rel_error,rmse\n";

    std::printf("%-10s %12s %12s %8s %12s %12s %12s\n",
                "N", "FP32(us)", "FP16(us)", "Speedup",
                "MaxRelErr", "MeanRelErr", "RMSE");
    std::printf("%-10s %12s %12s %8s %12s %12s %12s\n",
                "----------", "------------", "------------", "--------",
                "------------", "------------", "------------");
    std::fflush(stdout);

    for (int si = 0; si < nsizes; ++si) {
        long long n = sizes[si];
        std::printf("%-10lld ", n);
        std::fflush(stdout);

        // Generate signal (same for both precisions)
        unsigned seed = 42;
        std::srand(seed);
        std::vector<float> h_sig(n * 2);
        for (long long i = 0; i < n; ++i) {
            float t = (float)i / (float)n;
            h_sig[i * 2 + 0] = std::sin(2.0f * M_PI * 7.0f * t)
                             + 0.5f * std::sin(2.0f * M_PI * 23.0f * t)
                             + 0.25f * ((float)std::rand() / RAND_MAX - 0.5f);
            h_sig[i * 2 + 1] = 0.2f * std::cos(2.0f * M_PI * 13.0f * t)
                             + 0.1f * ((float)std::rand() / RAND_MAX - 0.5f);
        }

        // Normalize by 1/n to avoid FP16 overflow at large N
        float scale = 1.0f / (float)n;
        for (long long i = 0; i < n * 2; ++i)
            h_sig[i] *= scale;

        // ─── FP32 via Xt API (CUDA_C_32F) ───
        long long batch = 1;
        size_t f32_bytes = n * sizeof(cuComplex);

        // Convert float pairs to cuComplex
        std::vector<cuComplex> h_f32_in(n);
        for (long long i = 0; i < n; ++i)
            h_f32_in[i] = make_cuComplex(h_sig[i * 2], h_sig[i * 2 + 1]);

        cuComplex *d_f32_in = nullptr, *d_f32_out = nullptr;
        CHECK_CUDA(cudaMalloc(&d_f32_in, f32_bytes));
        CHECK_CUDA(cudaMalloc(&d_f32_out, f32_bytes));
        CHECK_CUDA(cudaMemcpy(d_f32_in, h_f32_in.data(), f32_bytes,
                              cudaMemcpyHostToDevice));

        // FP32 plan via Xt API
        cufftHandle plan_f32;
        CHECK_CUFFT(cufftCreate(&plan_f32));
        size_t ws_f32 = 0;
        long long n_xt = n;
        CHECK_CUFFT(cufftXtMakePlanMany(plan_f32, 1, &n_xt,
            nullptr, 1, n, CUDA_C_32F,
            nullptr, 1, n, CUDA_C_32F, 1, &ws_f32, CUDA_C_32F));

        void *d_work_f32 = nullptr;
        if (ws_f32 > 0) {
            CHECK_CUDA(cudaMalloc(&d_work_f32, ws_f32));
            CHECK_CUFFT(cufftXtSetWorkArea(plan_f32, &d_work_f32));
        }

        // Warmup
        for (int w = 0; w < WARMUP; ++w)
            CHECK_CUFFT(cufftXtExec(plan_f32, d_f32_in, d_f32_out,
                                    CUFFT_FORWARD));
        CHECK_CUDA(cudaDeviceSynchronize());

        // Benchmark FP32
        cudaEvent_t ev_start, ev_stop;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventRecord(ev_start));
        for (int i = 0; i < ITERS; ++i)
            CHECK_CUFFT(cufftXtExec(plan_f32, d_f32_in, d_f32_out,
                                    CUFFT_FORWARD));
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        float ms_f32 = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms_f32, ev_start, ev_stop));
        double us_f32 = ms_f32 * 1000.0 / ITERS;

        // Copy back
        std::vector<cuComplex> h_f32_out(n);
        CHECK_CUDA(cudaMemcpy(h_f32_out.data(), d_f32_out, f32_bytes,
                              cudaMemcpyDeviceToHost));

        // Cleanup FP32
        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_stop));
        CHECK_CUFFT(cufftDestroy(plan_f32));
        CHECK_CUDA(cudaFree(d_f32_in));
        CHECK_CUDA(cudaFree(d_f32_out));
        if (d_work_f32) CHECK_CUDA(cudaFree(d_work_f32));

        // ─── FP16 via Xt API (CUDA_C_16F) ───
        size_t f16_bytes = n * 2 * sizeof(__half);
        std::vector<__half> h_f16_in(n * 2);
        for (long long i = 0; i < n * 2; ++i)
            h_f16_in[i] = __float2half(h_sig[i]);

        __half *d_f16_in = nullptr, *d_f16_out = nullptr;
        CHECK_CUDA(cudaMalloc(&d_f16_in, f16_bytes));
        CHECK_CUDA(cudaMalloc(&d_f16_out, f16_bytes));
        CHECK_CUDA(cudaMemcpy(d_f16_in, h_f16_in.data(), f16_bytes,
                              cudaMemcpyHostToDevice));

        cufftHandle plan_f16;
        CHECK_CUFFT(cufftCreate(&plan_f16));
        size_t ws_f16 = 0;
        n_xt = n;
        CHECK_CUFFT(cufftXtMakePlanMany(plan_f16, 1, &n_xt,
            nullptr, 1, n, CUDA_C_16F,
            nullptr, 1, n, CUDA_C_16F, 1, &ws_f16, CUDA_C_16F));

        void *d_work_f16 = nullptr;
        if (ws_f16 > 0) {
            CHECK_CUDA(cudaMalloc(&d_work_f16, ws_f16));
            CHECK_CUFFT(cufftXtSetWorkArea(plan_f16, &d_work_f16));
        }

        for (int w = 0; w < WARMUP; ++w)
            CHECK_CUFFT(cufftXtExec(plan_f16, d_f16_in, d_f16_out,
                                    CUFFT_FORWARD));
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventRecord(ev_start));
        for (int i = 0; i < ITERS; ++i)
            CHECK_CUFFT(cufftXtExec(plan_f16, d_f16_in, d_f16_out,
                                    CUFFT_FORWARD));
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        float ms_f16 = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms_f16, ev_start, ev_stop));
        double us_f16 = ms_f16 * 1000.0 / ITERS;

        std::vector<__half> h_f16_out(n * 2);
        CHECK_CUDA(cudaMemcpy(h_f16_out.data(), d_f16_out, f16_bytes,
                              cudaMemcpyDeviceToHost));

        // ─── Compute error (FP16 vs FP32) ───
        double max_abs_err = 0, sum_rel = 0, sum_sq = 0;
        double f32_max_mag = 0;
        for (long long i = 0; i < n; ++i) {
            float r32 = cuCrealf(h_f32_out[i]);
            float i32 = cuCimagf(h_f32_out[i]);
            double mag = std::sqrt((double)r32 * r32 + (double)i32 * i32);
            if (mag > f32_max_mag) f32_max_mag = mag;
        }
        for (long long i = 0; i < n; ++i) {
            float r32 = cuCrealf(h_f32_out[i]);
            float i32 = cuCimagf(h_f32_out[i]);
            float r16 = __half2float(h_f16_out[i * 2 + 0]);
            float i16 = __half2float(h_f16_out[i * 2 + 1]);
            double diff = std::sqrt((double)(r32 - r16) * (r32 - r16)
                                  + (double)(i32 - i16) * (i32 - i16));
            double mag32 = std::sqrt((double)r32 * r32 + (double)i32 * i32);
            if (diff > max_abs_err) max_abs_err = diff;
            if (mag32 > 1e-10) sum_rel += diff / mag32;
            sum_sq += diff * diff;
        }
        double max_rel_err = max_abs_err;
        if (f32_max_mag > 1e-10) max_rel_err /= f32_max_mag;
        double mean_rel_err = sum_rel / n;
        double rmse = std::sqrt(sum_sq / n);

        double speedup = us_f32 / us_f16;

        std::printf("%12.2f %12.2f %7.2fx %12.2e %12.2e %12.2e\n",
                    us_f32, us_f16, speedup,
                    max_rel_err, mean_rel_err, rmse);
        std::fflush(stdout);

        csv << n << ","
            << std::fixed << std::setprecision(3) << us_f32 << ","
            << us_f16 << ","
            << std::setprecision(4) << speedup << ","
            << std::scientific << max_rel_err << ","
            << mean_rel_err << ","
            << rmse << "\n";

        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_stop));
        CHECK_CUFFT(cufftDestroy(plan_f16));
        CHECK_CUDA(cudaFree(d_f16_in));
        CHECK_CUDA(cudaFree(d_f16_out));
        if (d_work_f16) CHECK_CUDA(cudaFree(d_work_f16));
    }

    csv.close();
    std::printf("\nDone. Results saved to data/bench-fp32-vs-fp16-xt.csv\n");
    return 0;
}
