/**
 * FP8 E4M3 Verification — RTX 5070 Ti (Blackwell SM_120)
 *
 * Phase 3.1e: Minimal CUDA program to verify:
 *   1. __nv_fp8_e4m3 load/store basic operations
 *   2. FP8 ↔ FP32 roundtrip precision
 *   3. N=256 naive FP8 FFT SQNR measurement
 *
 * Compile: make (uses project Makefile, target: build/fp8_verification.exe)
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ─── Error check macros ───────────────────────────────────────────────
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t e = (call);                                       \
        if (e != cudaSuccess) {                                       \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(e));      \
            std::exit(1);                                             \
        }                                                             \
    } while (0)

// ─── Roundtrip helpers ─────────────────────────────────────────────────
__host__ __device__ inline __nv_fp8_e4m3 float_to_fp8(float x) {
    return __nv_fp8_e4m3(x);
}

__host__ __device__ inline float fp8_to_float(__nv_fp8_e4m3 x) {
    return float(x);
}

// ─── Kernel 1: Roundtrip precision test ────────────────────────────────
__global__ void kernel_roundtrip(const float *__restrict__ input,
                                  float *__restrict__ output,
                                  float *__restrict__ abs_err,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = input[idx];
    __nv_fp8_e4m3 f8 = float_to_fp8(val);
    float recovered = fp8_to_float(f8);
    output[idx] = recovered;
    abs_err[idx] = fabsf(val - recovered);
}

// ─── Kernel 2: FP8 load/store bandwidth test ───────────────────────────
__global__ void kernel_store_load(const float *__restrict__ input,
                                   float *__restrict__ output,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    __nv_fp8_e4m3 f8 = float_to_fp8(input[idx]);
    output[idx] = fp8_to_float(f8);
}

// ─── Kernel 3: Naive FP8 radix-2 DIT FFT, N=256 ────────────────────────
// Single-block implementation for simplicity (N=256 fits in shared mem).
// Every multiply and add is quantized to FP8 E4M3.
__global__ void kernel_fp8_fft_256(float *d_real, float *d_imag) {
    __shared__ float re[256];
    __shared__ float im[256];

    int tid = threadIdx.x;

    // Load + quantize to FP8
    __nv_fp8_e4m3 f8_re = float_to_fp8(d_real[tid]);
    __nv_fp8_e4m3 f8_im = float_to_fp8(d_imag[tid]);
    re[tid] = fp8_to_float(f8_re);
    im[tid] = fp8_to_float(f8_im);
    __syncthreads();

    // Bit-reversal permutation (precomputed offsets)
    // Using a small LUT for N=256
    unsigned int rev = 0;
    unsigned int t = tid;
    for (int b = 0; b < 8; b++) {
        rev = (rev << 1) | (t & 1);
        t >>= 1;
    }
    if (tid < rev) {
        float tmp = re[tid]; re[tid] = re[rev]; re[rev] = tmp;
        tmp = im[tid]; im[tid] = im[rev]; im[rev] = tmp;
    }
    __syncthreads();

    // Butterfly stages: 8 stages for N=256
    for (int stage_len = 1; stage_len < 256; stage_len <<= 1) {
        int jump = stage_len << 1;

        // Twiddle factor for this stage
        float angle = -M_PI / (float)stage_len;
        float w_re = cosf(angle);
        float w_im = sinf(angle);

        int group = (tid / stage_len) * jump;
        int pair_off = tid % stage_len;
        int a_idx = group + pair_off;
        int b_idx = a_idx + stage_len;

        // Load twiddle: W^k = (w_re + i*w_im)^k
        int k = pair_off;
        float tw_re = cosf(angle * k);
        float tw_im = sinf(angle * k);

        if (a_idx < 256 && b_idx < 256) {
            float ar = re[a_idx], ai = im[a_idx];
            float br = re[b_idx], bi = im[b_idx];

            // W × B (FP8 quantized)
            float wbr = tw_re * br - tw_im * bi;
            float wbi = tw_re * bi + tw_im * br;
            wbr = fp8_to_float(float_to_fp8(wbr));
            wbi = fp8_to_float(float_to_fp8(wbi));

            // A' = A + WB, B' = A - WB (FP8 quantized)
            float apr = fp8_to_float(float_to_fp8(ar + wbr));
            float api = fp8_to_float(float_to_fp8(ai + wbi));
            float bpr = fp8_to_float(float_to_fp8(ar - wbr));
            float bpi = fp8_to_float(float_to_fp8(ai - wbi));

            __syncthreads();
            re[a_idx] = apr; im[a_idx] = api;
            re[b_idx] = bpr; im[b_idx] = bpi;
            __syncthreads();
        }
        __syncthreads();
    }

    // Write back
    d_real[tid] = re[tid];
    d_imag[tid] = im[tid];
}

// ─── Host-side FP32 FFT reference (N=256) ──────────────────────────────
void cpu_fft_256(const float *in_re, const float *in_im,
                  float *out_re, float *out_im) {
    // Copy input
    float re[256], im[256];
    for (int i = 0; i < 256; i++) {
        re[i] = in_re[i];
        im[i] = in_im[i];
    }

    // Bit reversal
    for (int i = 1, j = 0; i < 256; i++) {
        int bit = 256 >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tmp = re[i]; re[i] = re[j]; re[j] = tmp;
            tmp = im[i]; im[i] = im[j]; im[j] = tmp;
        }
    }

    // Butterfly
    for (int len = 1; len < 256; len <<= 1) {
        int jump = len << 1;
        float angle = -M_PI / len;
        for (int g = 0; g < 256; g += jump) {
            for (int p = 0; p < len; p++) {
                float w_re = cosf(angle * p);
                float w_im = sinf(angle * p);
                int a = g + p, b = a + len;
                float tr = w_re * re[b] - w_im * im[b];
                float ti = w_re * im[b] + w_im * re[b];
                float ar = re[a], ai = im[a];
                re[a] = ar + tr; im[a] = ai + ti;
                re[b] = ar - tr; im[b] = ai - ti;
            }
        }
    }

    for (int i = 0; i < 256; i++) {
        out_re[i] = re[i];
        out_im[i] = im[i];
    }
}

// ─── Signal generators ─────────────────────────────────────────────────
void gen_chirp(float *re, float *im, int n) {
    for (int i = 0; i < n; i++) {
        float t = (float)i / (float)n;
        float phase = 2.0f * M_PI * 0.5f * t * t * n;
        re[i] = cosf(phase) / (float)n;  // normalize
        im[i] = sinf(phase) / (float)n;
    }
}

void gen_tones(float *re, float *im, int n, int n_tones) {
    for (int i = 0; i < n; i++) {
        re[i] = 0.0f;
        im[i] = 0.0f;
    }
    for (int t = 0; t < n_tones; t++) {
        int f = (t * 17 + 3) % n;
        float amp = 0.5f + 0.5f * ((float)(t + 1) / n_tones);
        float phase = (float)t * 1.7f;
        for (int i = 0; i < n; i++) {
            float phi = 2.0f * M_PI * f * i / n + phase;
            re[i] += amp * cosf(phi) / (float)n;
            im[i] += amp * sinf(phi) / (float)n;
        }
    }
}

// ─── SQNR computation ──────────────────────────────────────────────────
struct SnrResult {
    double snr_db;
    double max_abs_err;
    double rmse;
};

SnrResult compute_snr(const float *ref_re, const float *ref_im,
                       const float *test_re, const float *test_im, int n) {
    double sig_pow = 0.0, err_pow = 0.0;
    double max_err = 0.0;

    for (int i = 0; i < n; i++) {
        double dr = (double)ref_re[i] - (double)test_re[i];
        double di = (double)ref_im[i] - (double)test_im[i];
        double err = dr * dr + di * di;
        double sig = (double)ref_re[i] * ref_re[i] + (double)ref_im[i] * ref_im[i];
        err_pow += err;
        sig_pow += sig;
        double abs_err = sqrt(err);
        if (abs_err > max_err) max_err = abs_err;
    }

    SnrResult r;
    r.snr_db = 10.0 * log10(sig_pow / (err_pow + 1e-30));
    r.max_abs_err = max_err;
    r.rmse = sqrt(err_pow / n);
    return r;
}

// ─── Main ──────────────────────────────────────────────────────────────
int main() {
    // Device info
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    std::printf("=== FP8 E4M3 Verification ===\n");
    std::printf("Device: %s (SM %d.%d, %.1f GB VRAM)\n\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / 1073741824.0);

    const int BLOCK = 256;
    int grid;

    // ──────────────────────────────────────────────────────────────────
    // Test 1: FP8 ↔ FP32 roundtrip precision
    // ──────────────────────────────────────────────────────────────────
    {
        std::printf("--- Test 1: FP8 ↔ FP32 Roundtrip Precision ---\n");

        const int N = 1024 * 1024;  // 1M values
        std::vector<float> h_in(N), h_out(N), h_err(N);

        // Generate values covering FP8 range: subnormals, normals, max
        for (int i = 0; i < N; i++) {
            float t = (float)i / (float)(N - 1);
            // Log-uniform over FP8 representable range [2^-9, 448]
            float log_val = -9.0f + t * (log2f(448.0f) + 9.0f);
            float val = powf(2.0f, log_val);
            if (i < N / 2) val = -val;  // half negative
            h_in[i] = val;
        }

        float *d_in, *d_out, *d_err;
        CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_err, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(float),
                              cudaMemcpyHostToDevice));

        grid = (N + BLOCK - 1) / BLOCK;
        kernel_roundtrip<<<grid, BLOCK>>>(d_in, d_out, d_err, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_err.data(), d_err, N * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Stats
        double max_rel_err = 0.0, sum_rel_err = 0.0;
        int rel_count = 0;
        for (int i = 0; i < N; i++) {
            double abs_err = fabs((double)h_err[i]);
            double ref_val = fabs((double)h_in[i]);
            if (ref_val > 1e-8) {
                double rel = abs_err / ref_val;
                if (rel > max_rel_err) max_rel_err = rel;
                sum_rel_err += rel;
                rel_count++;
            }
        }
        double mean_rel_err = sum_rel_err / rel_count;

        std::printf("  Values tested: %d\n", N);
        std::printf("  Max relative error: %.4f (%.2f%%)\n",
                    max_rel_err, max_rel_err * 100.0);
        std::printf("  Mean relative error: %.4f (%.2f%%)\n",
                    mean_rel_err, mean_rel_err * 100.0);
        std::printf("  Theory (E4M3 ulp/2): 6.25%%\n");
        std::printf("  Roundtrip status: %s\n\n",
                    max_rel_err < 0.07 ? "PASS (within 7%%)" : "WARN");

        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_out));
        CHECK_CUDA(cudaFree(d_err));
    }

    // ──────────────────────────────────────────────────────────────────
    // Test 2: FP8 load/store bandwidth check
    // ──────────────────────────────────────────────────────────────────
    {
        std::printf("--- Test 2: FP8 Load/Store Basic Operations ---\n");

        const int N = 256;
        float h_in[256], h_out[256];
        for (int i = 0; i < N; i++) {
            h_in[i] = (float)i * 0.5f - 64.0f;
        }

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float),
                              cudaMemcpyHostToDevice));

        kernel_store_load<<<1, N>>>(d_in, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Verify a few values
        bool ok = true;
        for (int i = 0; i < N; i += 32) {
            float orig = h_in[i];
            float recovered = h_out[i];
            float rel = fabsf(orig - recovered) / (fabsf(orig) + 1e-10f);
            std::printf("  in=%.2f → fp8 → out=%.4f (rel=%.4f)\n",
                        orig, recovered, rel);
            if (rel > 0.07f && fabsf(orig) > 0.001f) ok = false;
        }
        std::printf("  Load/store: %s\n\n",
                    ok ? "PASS" : "FAIL (relative error > 7%%)");

        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_out));
    }

    // ──────────────────────────────────────────────────────────────────
    // Test 3: N=256 Naive FP8 FFT — SQNR measurement
    // ──────────────────────────────────────────────────────────────────
    {
        std::printf("--- Test 3: N=256 Naive FP8 FFT SQNR ---\n");

        const int N = 256;
        float h_re[N], h_im[N];

        // Test with chirp signal
        gen_chirp(h_re, h_im, N);

        // CPU reference FFT (FP32)
        float cpu_re[N], cpu_im[N];
        cpu_fft_256(h_re, h_im, cpu_re, cpu_im);

        // GPU FP8 FFT
        float *d_re, *d_im;
        CHECK_CUDA(cudaMalloc(&d_re, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_im, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_re, h_re, N * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_im, h_im, N * sizeof(float),
                              cudaMemcpyHostToDevice));

        kernel_fp8_fft_256<<<1, N>>>(d_re, d_im);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        float gpu_re[N], gpu_im[N];
        CHECK_CUDA(cudaMemcpy(gpu_re, d_re, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_im, d_im, N * sizeof(float),
                              cudaMemcpyDeviceToHost));

        SnrResult chirp_snr = compute_snr(cpu_re, cpu_im, gpu_re, gpu_im, N);
        std::printf("  Chirp signal: SNR = %.1f dB, max_err = %.4f, RMSE = %.6f\n",
                    chirp_snr.snr_db, chirp_snr.max_abs_err, chirp_snr.rmse);

        // Test with multitone signal
        gen_tones(h_re, h_im, N, 8);
        cpu_fft_256(h_re, h_im, cpu_re, cpu_im);

        CHECK_CUDA(cudaMemcpy(d_re, h_re, N * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_im, h_im, N * sizeof(float),
                              cudaMemcpyHostToDevice));

        kernel_fp8_fft_256<<<1, N>>>(d_re, d_im);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpy(gpu_re, d_re, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_im, d_im, N * sizeof(float),
                              cudaMemcpyDeviceToHost));

        SnrResult tone_snr = compute_snr(cpu_re, cpu_im, gpu_re, gpu_im, N);
        std::printf("  Multitone signal: SNR = %.1f dB, max_err = %.4f, RMSE = %.6f\n",
                    tone_snr.snr_db, tone_snr.max_abs_err, tone_snr.rmse);

        // Verdict
        double avg_snr = (chirp_snr.snr_db + tone_snr.snr_db) / 2.0;
        std::printf("\n  Average SQNR: %.1f dB\n", avg_snr);
        std::printf("  Paper (Bergach 2026) FP8 range: 14-20 dB\n");
        std::printf("  Python simulation (N=256): ~0 dB\n");

        if (avg_snr >= 10.0)
            std::printf("  Verdict: BETTER than expected (> 10 dB) — Blackwell FP8 may hit differently\n");
        else if (avg_snr >= 0.0)
            std::printf("  Verdict: MARGINAL — matches paper's \"FP8 is not practical for FFT\"\n");
        else
            std::printf("  Verdict: POOR — FP8 FFT without BFP is unusable\n");

        CHECK_CUDA(cudaFree(d_re));
        CHECK_CUDA(cudaFree(d_im));
    }

    std::printf("\n=== All FP8 verification tests complete ===\n");
    return 0;
}
