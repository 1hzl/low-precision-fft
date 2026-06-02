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

// ─── Config ─────────────────────────────────────────────────────
constexpr int WARMUP_ITERS = 10;
constexpr int BENCH_ITERS = 100;
constexpr unsigned long long SEED = 42;

// ─── Error check macros ─────────────────────────────────────────
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t e = (call);                                       \
        if (e != cudaSuccess) {                                       \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(e));      \
            std::exit(1);                                             \
        }                                                             \
    } while (0)

#define CHECK_CUFFT(call)                                             \
    do {                                                              \
        cufftResult r = (call);                                       \
        if (r != CUFFT_SUCCESS) {                                     \
            std::fprintf(stderr, "cuFFT error %s:%d: %d\n",          \
                    __FILE__, __LINE__, (int)r);                      \
            std::exit(1);                                             \
        }                                                             \
    } while (0)

// ─── Host input generation (same data for both precisions) ───────
std::vector<float> generate_signal(long long n, unsigned long long seed) {
    std::srand(seed);
    std::vector<float> h(n * 2);
    for (long long i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(n);
        h[i * 2 + 0] = std::sin(2.0f * M_PI * 7.0f * t)
                     + 0.5f * std::sin(2.0f * M_PI * 23.0f * t)
                     + 0.25f * ((float)std::rand() / RAND_MAX - 0.5f);
        h[i * 2 + 1] = 0.2f * std::cos(2.0f * M_PI * 13.0f * t)
                     + 0.1f * ((float)std::rand() / RAND_MAX - 0.5f);
    }
    return h;
}

// ─── FP32 FFT ────────────────────────────────────────────────────
double benchmark_fp32(long long n, const std::vector<float>& h_input,
                      std::vector<float>& h_output) {
    const long long batch = 1;
    const size_t data_bytes = n * sizeof(cuComplex);

    cuComplex *d_input = nullptr, *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, data_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, data_bytes));

    // Convert float pairs to cuComplex
    std::vector<cuComplex> h_input_cplx(n);
    for (long long i = 0; i < n; ++i)
        h_input_cplx[i] = make_cuComplex(h_input[i * 2], h_input[i * 2 + 1]);

    CHECK_CUDA(cudaMemcpy(d_input, h_input_cplx.data(), data_bytes,
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftCreate(&plan));
    CHECK_CUFFT(cufftPlan1d(&plan, (int)n, CUFFT_C2C, (int)batch));

    // Set stream to default
    CHECK_CUFFT(cufftSetStream(plan, 0));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i)
        CHECK_CUFFT(cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD));

    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; ++i)
        CHECK_CUFFT(cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double avg_us = (elapsed_ms * 1000.0) / BENCH_ITERS;

    // Copy result
    std::vector<cuComplex> h_out_cplx(n);
    CHECK_CUDA(cudaMemcpy(h_out_cplx.data(), d_output, data_bytes,
                          cudaMemcpyDeviceToHost));
    h_output.resize(n * 2);
    for (long long i = 0; i < n; ++i) {
        h_output[i * 2 + 0] = cuCrealf(h_out_cplx[i]);
        h_output[i * 2 + 1] = cuCimagf(h_out_cplx[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return avg_us;
}

// ─── FP16 FFT ────────────────────────────────────────────────────
double benchmark_fp16(long long n, const std::vector<float>& h_input,
                      std::vector<float>& h_output) {
    const long long batch = 1;
    const size_t data_half_bytes = n * 2 * sizeof(__half);

    // Convert float signal to half
    std::vector<__half> h_half_input(n * 2);
    for (long long i = 0; i < n * 2; ++i)
        h_half_input[i] = __float2half(h_input[i]);

    __half *d_input = nullptr, *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, data_half_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, data_half_bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_half_input.data(), data_half_bytes,
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftCreate(&plan));

    size_t workSize = 0;
    long long fft_n = n;

    CHECK_CUFFT(cufftXtMakePlanMany(
        plan, 1, &fft_n,
        nullptr, 1, n, CUDA_C_16F,       // input
        nullptr, 1, n, CUDA_C_16F,       // output
        batch, &workSize, CUDA_C_16F));   // exec type

    void *d_work = nullptr;
    if (workSize > 0) {
        CHECK_CUDA(cudaMalloc(&d_work, workSize));
        CHECK_CUFFT(cufftXtSetWorkArea(plan, &d_work));
    }

    CHECK_CUFFT(cufftSetStream(plan, 0));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i)
        CHECK_CUFFT(cufftXtExec(plan, d_input, d_output, CUFFT_FORWARD));

    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; ++i)
        CHECK_CUFFT(cufftXtExec(plan, d_input, d_output, CUFFT_FORWARD));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double avg_us = (elapsed_ms * 1000.0) / BENCH_ITERS;

    // Copy result and convert back to float
    std::vector<__half> h_half_output(n * 2);
    CHECK_CUDA(cudaMemcpy(h_half_output.data(), d_output, data_half_bytes,
                          cudaMemcpyDeviceToHost));
    h_output.resize(n * 2);
    for (long long i = 0; i < n * 2; ++i)
        h_output[i] = __half2float(h_half_output[i]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (d_work) CHECK_CUDA(cudaFree(d_work));
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return avg_us;
}

// ─── Error metrics ───────────────────────────────────────────────
struct ErrorMetrics {
    double max_rel_err;
    double mean_rel_err;
    double rmse;
};

ErrorMetrics compute_errors(const std::vector<float>& fp32,
                            const std::vector<float>& fp16, long long n) {
    ErrorMetrics m = {0.0, 0.0, 0.0};
    double sum_rel = 0.0, sum_sq = 0.0;
    double max_abs_err = 0.0;

    // Find max magnitude in FP32 for normalization
    double fp32_max_mag = 0.0;
    for (long long i = 0; i < n; ++i) {
        float re = fp32[i * 2], im = fp32[i * 2 + 1];
        double mag = std::sqrt((double)re * re + (double)im * im);
        if (mag > fp32_max_mag) fp32_max_mag = mag;
    }
    double noise_floor = fp32_max_mag * 1e-6;

    long long rel_count = 0;
    for (long long i = 0; i < n; ++i) {
        float r32 = fp32[i * 2], i32 = fp32[i * 2 + 1];
        float r16 = fp16[i * 2], i16 = fp16[i * 2 + 1];
        double abs_err = std::fabs((double)r32 - r16);
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        abs_err = std::fabs((double)i32 - i16);
        if (abs_err > max_abs_err) max_abs_err = abs_err;

        double mag32 = std::sqrt((double)r32 * r32 + (double)i32 * i32);
        double diff = std::sqrt((double)(r32 - r16) * (r32 - r16)
                              + (double)(i32 - i16) * (i32 - i16));
        if (mag32 > noise_floor) {
            sum_rel += diff / mag32;
            ++rel_count;
        }
        sum_sq += diff * diff;
    }
    m.max_rel_err = max_abs_err;
    if (fp32_max_mag > 1e-10)
        m.max_rel_err /= fp32_max_mag;
    m.mean_rel_err = rel_count > 0 ? sum_rel / rel_count : 0.0;
    m.rmse = std::sqrt(sum_sq / n);

    return m;
}

// ─── CSV output ──────────────────────────────────────────────────
void write_csv(const std::string& path,
               const std::vector<long long>& sizes,
               const std::vector<double>& us_fp32,
               const std::vector<double>& us_fp16,
               const std::vector<ErrorMetrics>& errors) {
    std::ofstream f(path);
    f << "n,fp32_us,fp16_us,speedup,max_rel_error,mean_rel_error,rmse\n";
    for (size_t i = 0; i < sizes.size(); ++i) {
        double speedup = us_fp32[i] / us_fp16[i];
        f << sizes[i] << ","
          << std::fixed << std::setprecision(3) << us_fp32[i] << ","
          << us_fp16[i] << ","
          << std::setprecision(4) << speedup << ","
          << std::scientific << errors[i].max_rel_err << ","
          << errors[i].mean_rel_err << ","
          << errors[i].rmse << "\n";
    }
}

// ─── Main ────────────────────────────────────────────────────────
int main() {
    const std::vector<long long> fft_sizes = {
        256, 512, 1024, 2048, 4096, 8192, 16384,
        32768, 65536, 131072, 262144, 524288, 1048576
    };

    std::printf("============================================================\n");
    std::printf("  cuFFT FP32 vs FP16 Benchmark\n");
    std::printf("  GPU: RTX 5070 Ti Laptop (SM 12.0)\n");
    std::printf("  CUDA: 13.3\n");
    std::printf("  Warmup: %d iters | Bench: %d iters\n", WARMUP_ITERS, BENCH_ITERS);
    std::printf("============================================================\n\n");

    // Print CUDA device info
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    std::printf("Device: %s (%.1f GB VRAM)\n\n", prop.name,
                prop.totalGlobalMem / 1073741824.0);

    std::printf("%-10s %12s %12s %8s %12s %12s %12s\n",
                "N", "FP32(us)", "FP16(us)", "Speedup",
                "MaxRelErr", "MeanRelErr", "RMSE");
    std::printf("%-10s %12s %12s %8s %12s %12s %12s\n",
                "----------", "------------", "------------", "--------",
                "------------", "------------", "------------");

    std::vector<double> all_us_fp32, all_us_fp16;
    std::vector<ErrorMetrics> all_errors;

    for (auto n : fft_sizes) {
        // Generate signal, normalize by 1/n to avoid FP16 overflow at large N
        auto h_input_raw = generate_signal(n, SEED);
        float scale = 1.0f / (float)n;
        auto h_input = std::vector<float>(h_input_raw.size());
        for (size_t i = 0; i < h_input_raw.size(); ++i)
            h_input[i] = h_input_raw[i] * scale;

        // FP32 benchmark
        std::vector<float> h_fp32_out;
        double us_fp32 = benchmark_fp32(n, h_input, h_fp32_out);

        // FP16 benchmark
        std::vector<float> h_fp16_out;
        double us_fp16 = benchmark_fp16(n, h_input, h_fp16_out);

        // Error
        ErrorMetrics err = compute_errors(h_fp32_out, h_fp16_out, n);

        all_us_fp32.push_back(us_fp32);
        all_us_fp16.push_back(us_fp16);
        all_errors.push_back(err);

        double speedup = us_fp32 / us_fp16;

        std::printf("%-10lld %12.2f %12.2f %7.2fx %12.2e %12.2e %12.2e\n",
                    n, us_fp32, us_fp16, speedup,
                    err.max_rel_err, err.mean_rel_err, err.rmse);
    }

    // Write CSV
    write_csv("data/bench-fp32-vs-fp16.csv", fft_sizes,
              all_us_fp32, all_us_fp16, all_errors);
    std::printf("\nResults saved to data/bench-fp32-vs-fp16.csv\n");

    // Summary
    double avg_speedup = 0;
    double max_speedup = 0;
    double worst_err = 0;
    long long worst_n = 0;
    for (size_t i = 0; i < all_us_fp32.size(); ++i) {
        double sp = all_us_fp32[i] / all_us_fp16[i];
        avg_speedup += sp;
        max_speedup = std::max(max_speedup, sp);
        if (all_errors[i].max_rel_err > worst_err) {
            worst_err = all_errors[i].max_rel_err;
            worst_n = fft_sizes[i];
        }
    }
    avg_speedup /= fft_sizes.size();

    std::printf("\n--- Summary ---\n");
    std::printf("Average speedup: %.2fx\n", avg_speedup);
    std::printf("Max speedup:     %.2fx\n", max_speedup);
    std::printf("Worst rel error: %.2e (n=%lld)\n", worst_err, worst_n);
    std::printf("All max_rel_err < 1e-3: %s\n",
                worst_err < 1e-3 ? "PASS" : "FAIL");

    return 0;
}
