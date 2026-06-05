/**
 * bfp_fft.cu — Block Floating-Point Radix-2 DIT FFT (Sprint 3.3)
 *
 * Per-stage BFP: each FFT stage shares one integer exponent.
 * Dequantize FP8 mantissas → butterfly in float32 → requantize to FP8.
 *
 * Target: RTX 5070 Ti (SM_120, Blackwell), CUDA 13.3
 *
 * Build: nvcc -arch=sm_120 -O3 -o build/bfp_fft.exe src/cuda/bfp_fft.cu
 */

#define BFP_FFT_EXPORT
#include "bfp_fft.h"

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define FP8_MAX 448.0f
#define BLOCK_SIZE 256

// ─── Error checking ────────────────────────────────────────────────────

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t e = (call);                                       \
        if (e != cudaSuccess) {                                       \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(e));      \
            std::exit(1);                                             \
        }                                                             \
    } while (0)

// ─── FP8 conversion helpers ────────────────────────────────────────────

__host__ __device__ inline __nv_fp8_e4m3 float_to_fp8(float x) {
    return __nv_fp8_e4m3(x);
}

__host__ __device__ inline float fp8_to_float(__nv_fp8_e4m3 x) {
    return float(x);
}

// ─── Bit reversal ──────────────────────────────────────────────────────

static int bit_reverse(int i, int log2N) {
    int rev = 0;
    for (int b = 0; b < log2N; b++) {
        rev = (rev << 1) | (i & 1);
        i >>= 1;
    }
    return rev;
}

// ─── Exponent computation ──────────────────────────────────────────────

static __host__ __device__ int compute_exponent_from_max(float max_val) {
    if (max_val == 0.0f) return 0;
    float f_exp = log2f(max_val / FP8_MAX);
    int E = (int)floorf(f_exp);
    // Clamp: if mantissa would exceed FP8_MAX, bump exponent
    while (max_val / exp2f((float)E) > FP8_MAX) {
        E++;
    }
    return E;
}

// ─── Kernel: BFP FFT DIT stage (dequant + butterfly + atomicMax) ──────

__global__ void bfp_fft_dit_stage(
    const __nv_fp8_e4m3* __restrict__ fp8_re,
    const __nv_fp8_e4m3* __restrict__ fp8_im,
    float* __restrict__ work_re,
    float* __restrict__ work_im,
    unsigned int* __restrict__ max_abs_bits,
    const int* __restrict__ stages_exp,
    int stage,
    int step,
    int N,
    int inverse
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pairs = N / 2;
    if (tid >= num_pairs) return;

    int jump = step << 1;
    int pair_off = tid % step;
    int group_start = (tid / step) * jump;
    int a = group_start + pair_off;
    int b = a + step;

    int shared_exp = stages_exp[stage];
    float scale = exp2f((float)shared_exp);

    // Dequantize
    float ar = fp8_to_float(fp8_re[a]) * scale;
    float ai = fp8_to_float(fp8_im[a]) * scale;
    float br = fp8_to_float(fp8_re[b]) * scale;
    float bi = fp8_to_float(fp8_im[b]) * scale;

    // Twiddle W^k = exp(sign * pi * i * pair_off / step)
    float angle_sign = inverse ? M_PI : -M_PI;
    float angle = angle_sign * (float)pair_off / (float)step;
    float tw_re = cosf(angle);
    float tw_im = sinf(angle);

    // Complex multiply: W * B
    float wbr = tw_re * br - tw_im * bi;
    float wbi = tw_re * bi + tw_im * br;

    // Butterfly: A' = A + WB, B' = A - WB
    float apr = ar + wbr;
    float api = ai + wbi;
    float bpr = ar - wbr;
    float bpi = ai - wbi;

    // Write to float workspace
    work_re[a] = apr;
    work_im[a] = api;
    work_re[b] = bpr;
    work_im[b] = bpi;

    // Track max absolute value for next stage's shared exponent
    // atomicMax on unsigned int bit pattern works for non-negative floats
    unsigned int u_apr = __float_as_uint(fabsf(apr));
    unsigned int u_api = __float_as_uint(fabsf(api));
    unsigned int u_bpr = __float_as_uint(fabsf(bpr));
    unsigned int u_bpi = __float_as_uint(fabsf(bpi));

    atomicMax(max_abs_bits, u_apr);
    atomicMax(max_abs_bits, u_api);
    atomicMax(max_abs_bits, u_bpr);
    atomicMax(max_abs_bits, u_bpi);
}

// ─── Kernel: Requantize float workspace to FP8 mantissas ───────────────

__global__ void bfp_requantize(
    const float* __restrict__ work_re,
    const float* __restrict__ work_im,
    __nv_fp8_e4m3* __restrict__ fp8_re,
    __nv_fp8_e4m3* __restrict__ fp8_im,
    const unsigned int* __restrict__ stage_max_bits,
    int* __restrict__ stages_exp,
    int stage,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Thread 0 computes shared exponent from stage atomicMax result
    __shared__ int s_exp;
    if (threadIdx.x == 0) {
        float stage_max;
        unsigned int bits = *stage_max_bits;
        // Interpret unsigned int bits as float (works for non-negative floats)
        stage_max = __uint_as_float(bits);
        s_exp = compute_exponent_from_max(stage_max);
        stages_exp[stage] = s_exp;
    }
    __syncthreads();

    float inv_scale = exp2f(-(float)s_exp);
    fp8_re[idx] = float_to_fp8(work_re[idx] * inv_scale);
    fp8_im[idx] = float_to_fp8(work_im[idx] * inv_scale);
}

// ─── Kernel: Final dequantize FP8 → float output ───────────────────────

__global__ void bfp_dequant_output(
    const __nv_fp8_e4m3* __restrict__ fp8_re,
    const __nv_fp8_e4m3* __restrict__ fp8_im,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    const int* __restrict__ stages_exp,
    int stage,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int shared_exp = stages_exp[stage];
    float scale = exp2f((float)shared_exp);
    out_re[idx] = fp8_to_float(fp8_re[idx]) * scale;
    out_im[idx] = fp8_to_float(fp8_im[idx]) * scale;
}

// ─── Kernel: Scale output by 1/N (inverse FFT only) ──────────────────────

__global__ void bfp_scale_output(
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float inv_n = 1.0f / (float)N;
    out_re[idx] *= inv_n;
    out_im[idx] *= inv_n;
}

// ─── Host: Shared BFP FFT helper ──────────────────────────────────────

static int bfp_fft_run(
    const float* x_real, const float* x_imag,
    float* X_real, float* X_imag,
    int N,
    int* stages_exp,
    int inverse
) {
    if (N < 2 || (N & (N - 1)) != 0) {
        std::fprintf(stderr, "bfp_fft: N=%d must be power of 2 and >= 2\n", N);
        return -1;
    }
    int log2N = 0;
    for (int t = N; t > 1; t >>= 1) log2N++;

    // ── 1. Bit-reverse + initial quantization on host ─────────────────
    std::vector<__nv_fp8_e4m3> h_fp8_re(N), h_fp8_im(N);

    // Compute initial shared exponent from (bit-reversed) input
    float max_abs = 0.0f;
    for (int i = 0; i < N; i++) {
        int rev = bit_reverse(i, log2N);
        float a = fabsf(x_real[rev]);
        float b = fabsf(x_imag[rev]);
        if (a > max_abs) max_abs = a;
        if (b > max_abs) max_abs = b;
    }
    int E_cur = compute_exponent_from_max(max_abs);
    stages_exp[0] = E_cur;

    float init_scale = exp2f(-(float)E_cur);
    for (int i = 0; i < N; i++) {
        int rev = bit_reverse(i, log2N);
        h_fp8_re[i] = float_to_fp8(x_real[rev] * init_scale);
        h_fp8_im[i] = float_to_fp8(x_imag[rev] * init_scale);
    }

    // ── 2. Allocate device memory ────────────────────────────────────
    __nv_fp8_e4m3 *d_fp8_re, *d_fp8_im;
    float *d_work_re, *d_work_im;
    unsigned int *d_stage_max_bits;
    int *d_stages_exp;

    CHECK_CUDA(cudaMalloc(&d_fp8_re, N * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_fp8_im, N * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_work_re, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_work_im, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_stage_max_bits, (log2N + 1) * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_stages_exp, (log2N + 1) * sizeof(int)));

    // Zero-initialize per-stage max bits and exponents
    CHECK_CUDA(cudaMemset(d_stage_max_bits, 0, (log2N + 1) * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_stages_exp, 0, (log2N + 1) * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_fp8_re, h_fp8_re.data(),
                           N * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_fp8_im, h_fp8_im.data(),
                           N * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));

    // Write initial exponent (stage 0) to device
    int host_E0 = E_cur;
    CHECK_CUDA(cudaMemcpy(d_stages_exp, &host_E0, sizeof(int),
                           cudaMemcpyHostToDevice));

    int grid_pairs = (N / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_full  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ── 3. Stage-by-stage butterfly (all GPU, no host sync) ──────────
    for (int s = 0; s < log2N; s++) {
        int step = 1 << s;

        // Kernel 1: dequant → butterfly → float workspace + atomicMax
        bfp_fft_dit_stage<<<grid_pairs, BLOCK_SIZE>>>(
            d_fp8_re, d_fp8_im, d_work_re, d_work_im,
            d_stage_max_bits + s + 1,  // per-stage max output
            d_stages_exp, s,           // read shared_exp from previous stage
            step, N, inverse);
        CHECK_CUDA(cudaGetLastError());

        // Kernel 2: compute exponent from max + requantize → FP8
        bfp_requantize<<<grid_full, BLOCK_SIZE>>>(
            d_work_re, d_work_im, d_fp8_re, d_fp8_im,
            d_stage_max_bits + s + 1,  // read per-stage max
            d_stages_exp, s + 1,       // write computed exponent
            N);
        CHECK_CUDA(cudaGetLastError());
    }

    // Single sync after all stages — catches any accumulated kernel errors
    CHECK_CUDA(cudaDeviceSynchronize());

    // Retrieve exponents from device
    CHECK_CUDA(cudaMemcpy(stages_exp, d_stages_exp, (log2N + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost));

    // ── 4. Final dequantize → float output ──────────────────────────
    bfp_dequant_output<<<grid_full, BLOCK_SIZE>>>(
        d_fp8_re, d_fp8_im, d_work_re, d_work_im,
        d_stages_exp, log2N,  // read final exponent from device
        N);
    CHECK_CUDA(cudaGetLastError());

    if (inverse) {
        bfp_scale_output<<<grid_full, BLOCK_SIZE>>>(
            d_work_re, d_work_im, N);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaMemcpy(X_real, d_work_re, N * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(X_imag, d_work_im, N * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // ── 5. Cleanup ──────────────────────────────────────────────────
    CHECK_CUDA(cudaFree(d_fp8_re));
    CHECK_CUDA(cudaFree(d_fp8_im));
    CHECK_CUDA(cudaFree(d_work_re));
    CHECK_CUDA(cudaFree(d_work_im));
    CHECK_CUDA(cudaFree(d_stage_max_bits));
    CHECK_CUDA(cudaFree(d_stages_exp));

    return 0;
}

// ─── Host: Full BFP forward FFT ────────────────────────────────────────

BFP_API int bfp_fft_forward(
    const float* x_real, const float* x_imag,
    float* X_real, float* X_imag,
    int N,
    int* stages_exp
) {
    return bfp_fft_run(x_real, x_imag, X_real, X_imag, N, stages_exp, 0);
}

// ─── Host: Full BFP inverse FFT ────────────────────────────────────────

BFP_API int bfp_fft_inverse(
    const float* x_real, const float* x_imag,
    float* X_real, float* X_imag,
    int N,
    int* stages_exp
) {
    return bfp_fft_run(x_real, x_imag, X_real, X_imag, N, stages_exp, 1);
}

// ─── SQNR computation ──────────────────────────────────────────────────

BFP_API double bfp_compute_sqnr(
    const float* ref_real, const float* ref_imag,
    const float* tst_real, const float* tst_imag,
    int N
) {
    double sig_pow = 0.0, err_pow = 0.0;
    for (int i = 0; i < N; i++) {
        double dr = (double)ref_real[i] - (double)tst_real[i];
        double di = (double)ref_imag[i] - (double)tst_imag[i];
        err_pow += dr * dr + di * di;
        sig_pow += (double)ref_real[i] * ref_real[i]
                 + (double)ref_imag[i] * ref_imag[i];
    }
    return 10.0 * log10(sig_pow / (err_pow + 1e-30));
}

// ─── CPU reference FFT ─────────────────────────────────────────────────

static void cpu_fft_radix2(
    const float* in_re, const float* in_im,
    float* out_re, float* out_im,
    int N, bool inverse)
{
    // Copy
    for (int i = 0; i < N; i++) {
        out_re[i] = in_re[i];
        out_im[i] = in_im[i];
    }

    // Bit reversal
    int log2N = 0;
    for (int t = N; t > 1; t >>= 1) log2N++;

    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tmp = out_re[i]; out_re[i] = out_re[j]; out_re[j] = tmp;
            tmp = out_im[i]; out_im[i] = out_im[j]; out_im[j] = tmp;
        }
    }

    // Butterfly stages
    int step = 1;
    while (step < N) {
        int jump = step << 1;
        float angle_sign = inverse ? M_PI : -M_PI;
        for (int g = 0; g < N; g += jump) {
            for (int p = 0; p < step; p++) {
                float ang = angle_sign * (float)p / (float)step;
                float w_re = cosf(ang);
                float w_im = sinf(ang);
                int a = g + p;
                int b = a + step;
                float tr = w_re * out_re[b] - w_im * out_im[b];
                float ti = w_re * out_im[b] + w_im * out_re[b];
                float ar = out_re[a], ai = out_im[a];
                out_re[a] = ar + tr; out_im[a] = ai + ti;
                out_re[b] = ar - tr; out_im[b] = ai - ti;
            }
        }
        step = jump;
    }

    if (inverse) {
        float inv_n = 1.0f / (float)N;
        for (int i = 0; i < N; i++) {
            out_re[i] *= inv_n;
            out_im[i] *= inv_n;
        }
    }
}

// ─── Signal generators ─────────────────────────────────────────────────

static void gen_chirp(float* re, float* im, int N) {
    for (int i = 0; i < N; i++) {
        float t = (float)i / (float)N;
        float phase = 2.0f * M_PI * 0.5f * t * t * (float)N;
        // Normalized: amplitude 1/N to prevent overflow growth
        re[i] = cosf(phase) / (float)N;
        im[i] = sinf(phase) / (float)N;
    }
}

static void gen_random_normal(float* re, float* im, int N, unsigned int seed) {
    // Simple LCG for reproducibility
    unsigned int state = seed;
    for (int i = 0; i < N; i++) {
        state = state * 1103515245 + 12345;
        // Box-Muller transform
        float u1 = (float)(state & 0x7FFFFFFF) / 2147483648.0f;
        state = state * 1103515245 + 12345;
        float u2 = (float)(state & 0x7FFFFFFF) / 2147483648.0f;
        float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
        float theta = 2.0f * M_PI * u2;
        // Normalize by N to prevent FFT overflow
        re[i] = r * cosf(theta) / (float)N / 3.0f;
        im[i] = r * sinf(theta) / (float)N / 3.0f;
    }
}

// ─── GPU Benchmark ─────────────────────────────────────────────────────

static double run_bfp_gpu_benchmark(int N, int warmup, int reps, int batch) {
    int log2N = 0;
    for (int t = N; t > 1; t >>= 1) log2N++;
    int grid_pairs = (N / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_full  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Generate random normal test data
    std::vector<float> h_x_re(N), h_x_im(N);
    gen_random_normal(h_x_re.data(), h_x_im.data(), N, 42);

    // Bit-reverse + initial quantization on host
    std::vector<__nv_fp8_e4m3> h_fp8_re(N), h_fp8_im(N);
    float max_abs = 0.0f;
    for (int i = 0; i < N; i++) {
        int rev = bit_reverse(i, log2N);
        float a = fabsf(h_x_re[rev]), b = fabsf(h_x_im[rev]);
        if (a > max_abs) max_abs = a;
        if (b > max_abs) max_abs = b;
    }
    int E0 = compute_exponent_from_max(max_abs);
    float init_scale = exp2f(-(float)E0);
    for (int i = 0; i < N; i++) {
        int rev = bit_reverse(i, log2N);
        h_fp8_re[i] = float_to_fp8(h_x_re[rev] * init_scale);
        h_fp8_im[i] = float_to_fp8(h_x_im[rev] * init_scale);
    }

    // Allocate device memory (work + pristine copies for reset)
    __nv_fp8_e4m3 *d_fp8_re, *d_fp8_im;
    __nv_fp8_e4m3 *d_fp8_re_init, *d_fp8_im_init;
    float *d_work_re, *d_work_im;
    unsigned int *d_stage_max_bits;
    int *d_stages_exp;

    CHECK_CUDA(cudaMalloc(&d_fp8_re, N * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_fp8_im, N * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_fp8_re_init, N * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_fp8_im_init, N * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_work_re, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_work_im, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_stage_max_bits, (log2N + 1) * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_stages_exp, (log2N + 1) * sizeof(int)));

    // Copy pristine initial data to device
    CHECK_CUDA(cudaMemcpy(d_fp8_re_init, h_fp8_re.data(),
                           N * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_fp8_im_init, h_fp8_im.data(),
                           N * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));

    // CUDA events for timing
    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    std::vector<float> elapsed_ms(reps * batch);

    for (int rep = 0; rep < reps; rep++) {
        for (int b = 0; b < batch; b++) {
            // Reset working buffers from pristine copies
            CHECK_CUDA(cudaMemcpy(d_fp8_re, d_fp8_re_init,
                                   N * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(d_fp8_im, d_fp8_im_init,
                                   N * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemset(d_stage_max_bits, 0, (log2N + 1) * sizeof(unsigned int)));
            CHECK_CUDA(cudaMemset(d_stages_exp, 0, (log2N + 1) * sizeof(int)));
            int host_E0 = E0;
            CHECK_CUDA(cudaMemcpy(d_stages_exp, &host_E0, sizeof(int),
                                   cudaMemcpyHostToDevice));

            // Time GPU kernel execution
            CHECK_CUDA(cudaEventRecord(ev_start));

            for (int s = 0; s < log2N; s++) {
                int step = 1 << s;
                bfp_fft_dit_stage<<<grid_pairs, BLOCK_SIZE>>>(
                    d_fp8_re, d_fp8_im, d_work_re, d_work_im,
                    d_stage_max_bits + s + 1,
                    d_stages_exp, s, step, N, 0);
                CHECK_CUDA(cudaGetLastError());

                bfp_requantize<<<grid_full, BLOCK_SIZE>>>(
                    d_work_re, d_work_im, d_fp8_re, d_fp8_im,
                    d_stage_max_bits + s + 1,
                    d_stages_exp, s + 1, N);
                CHECK_CUDA(cudaGetLastError());
            }

            bfp_dequant_output<<<grid_full, BLOCK_SIZE>>>(
                d_fp8_re, d_fp8_im, d_work_re, d_work_im,
                d_stages_exp, log2N, N);
            CHECK_CUDA(cudaGetLastError());

            CHECK_CUDA(cudaEventRecord(ev_stop));
            CHECK_CUDA(cudaEventSynchronize(ev_stop));

            float ms;
            CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            elapsed_ms[rep * batch + b] = ms;
        }
    }

    // Sort and find median
    std::sort(elapsed_ms.begin(), elapsed_ms.end());
    double median_ms = elapsed_ms[elapsed_ms.size() / 2];

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    CHECK_CUDA(cudaFree(d_fp8_re));
    CHECK_CUDA(cudaFree(d_fp8_im));
    CHECK_CUDA(cudaFree(d_fp8_re_init));
    CHECK_CUDA(cudaFree(d_fp8_im_init));
    CHECK_CUDA(cudaFree(d_work_re));
    CHECK_CUDA(cudaFree(d_work_im));
    CHECK_CUDA(cudaFree(d_stage_max_bits));
    CHECK_CUDA(cudaFree(d_stages_exp));

    return median_ms * 1000.0;  // ms → μs
}

// ─── Self-test main ────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // Device info
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    // ── Benchmark mode: --bench N [warmup] [reps] [batch] ────────────
    if (argc >= 3 && strcmp(argv[1], "--bench") == 0) {
        int N = atoi(argv[2]);
        int warmup = (argc > 3) ? atoi(argv[3]) : 50;
        int reps   = (argc > 4) ? atoi(argv[4]) : 200;
        int batch  = (argc > 5) ? atoi(argv[5]) : 1;

        // Warmup (discard)
        run_bfp_gpu_benchmark(N, 0, warmup, batch);

        // Timed run
        double median_us = run_bfp_gpu_benchmark(N, 0, reps, batch);
        double per_fft_us = median_us / (double)batch;

        std::printf("BENCH N=%d warmup=%d reps=%d batch=%d median_us=%.2f per_fft_us=%.2f\n",
                    N, warmup, reps, batch, median_us, per_fft_us);
        return 0;
    }

    // ── Benchmark-list mode: --bench-list [warmup] [reps] ────────────
    if (argc >= 2 && strcmp(argv[1], "--bench-list") == 0) {
        int warmup = (argc > 2) ? atoi(argv[2]) : 50;
        int reps   = (argc > 3) ? atoi(argv[3]) : 200;

        std::printf("# BFP FFT GPU Benchmark (Sprint 3.4)\n");
        std::printf("# Device: %s (SM %d.%d, CUDA %d)\n",
                    prop.name, prop.major, prop.minor,
                    CUDART_VERSION / 1000);
        std::printf("# %-6s  %-12s\n", "N", "median_us");
        std::printf("# --------------------\n");

        int N_values[] = {256, 512, 1024, 2048, 4096};
        int num_n = sizeof(N_values) / sizeof(N_values[0]);

        for (int ni = 0; ni < num_n; ni++) {
            int N = N_values[ni];

            // Warmup
            run_bfp_gpu_benchmark(N, 0, warmup, 1);

            // Timed
            double median_us = run_bfp_gpu_benchmark(N, 0, reps, 1);

            std::printf("BENCH N=%-6d median_us=%.2f\n", N, median_us);
        }
        std::printf("# Done.\n");
        return 0;
    }

    // ── Default: self-test mode ─────────────────────────────────────
    int test_N = 256;
    if (argc > 1) test_N = atoi(argv[1]);

    std::printf("# BFP FFT CUDA Kernel v0 (Sprint 3.3)\n");
    std::printf("# Device: %s (SM %d.%d, CUDA %d)\n",
                prop.name, prop.major, prop.minor,
                CUDART_VERSION / 1000);

    // Test sizes
    int N_values[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int num_n = sizeof(N_values) / sizeof(N_values[0]);

    std::printf("# %-6s  %-12s  %-12s  %-12s\n",
                "N", "Signal", "SQNR(dB)", "MaxAbsErr");
    std::printf("# -----------------------------------------------------------------\n");

    for (int ni = 0; ni < num_n; ni++) {
        int N = N_values[ni];
        if (argc > 1 && N != test_N) continue;

        int log2N = 0;
        for (int t = N; t > 1; t >>= 1) log2N++;

        // Allocate host arrays
        std::vector<float> x_re(N), x_im(N);
        std::vector<float> ref_re(N), ref_im(N);
        std::vector<float> bfp_re(N), bfp_im(N);
        std::vector<int> stages_exp(log2N + 1);

        // Test signal 1: chirp
        gen_chirp(x_re.data(), x_im.data(), N);
        cpu_fft_radix2(x_re.data(), x_im.data(), ref_re.data(), ref_im.data(), N, false);

        int ret = bfp_fft_forward(x_re.data(), x_im.data(),
                                   bfp_re.data(), bfp_im.data(),
                                   N, stages_exp.data());
        if (ret != 0) {
            std::printf("  %-6d  %-12s  ERROR\n", N, "chirp");
            break;
        }
        double sqnr_chirp = bfp_compute_sqnr(ref_re.data(), ref_im.data(),
                                               bfp_re.data(), bfp_im.data(), N);
        // Max abs error
        double max_err = 0.0;
        for (int i = 0; i < N; i++) {
            double dr = ref_re[i] - bfp_re[i];
            double di = ref_im[i] - bfp_im[i];
            double e = sqrt(dr*dr + di*di);
            if (e > max_err) max_err = e;
        }
        std::printf("  %-6d  %-12s  %8.1f dB   %.4e\n",
                    N, "chirp", sqnr_chirp, max_err);

        // Test signal 2: random normal
        gen_random_normal(x_re.data(), x_im.data(), N, 42 + N);
        cpu_fft_radix2(x_re.data(), x_im.data(), ref_re.data(), ref_im.data(), N, false);

        ret = bfp_fft_forward(x_re.data(), x_im.data(),
                                bfp_re.data(), bfp_im.data(),
                                N, stages_exp.data());
        if (ret != 0) {
            std::printf("  %-6d  %-12s  ERROR\n", N, "random");
            break;
        }
        double sqnr_rand = bfp_compute_sqnr(ref_re.data(), ref_im.data(),
                                              bfp_re.data(), bfp_im.data(), N);
        max_err = 0.0;
        for (int i = 0; i < N; i++) {
            double dr = ref_re[i] - bfp_re[i];
            double di = ref_im[i] - bfp_im[i];
            double e = sqrt(dr*dr + di*di);
            if (e > max_err) max_err = e;
        }
        std::printf("  %-6d  %-12s  %8.1f dB   %.4e\n",
                    N, "random", sqnr_rand, max_err);

        // Test signal 3: roundtrip (FFT → IFFT)
        std::vector<float> rt_re(N), rt_im(N);
        std::vector<int> ifft_exp(log2N + 1);
        std::vector<float> bfp_fft_re(N), bfp_fft_im(N);

        // Forward FFT first
        bfp_fft_forward(x_re.data(), x_im.data(),
                        bfp_fft_re.data(), bfp_fft_im.data(),
                        N, stages_exp.data());

        // Then inverse FFT
        ret = bfp_fft_inverse(bfp_fft_re.data(), bfp_fft_im.data(),
                              rt_re.data(), rt_im.data(),
                              N, ifft_exp.data());
        if (ret != 0) {
            std::printf("  %-6d  %-12s  ERROR\n", N, "roundtrip");
            break;
        }
        double sqnr_rt = bfp_compute_sqnr(x_re.data(), x_im.data(),
                                           rt_re.data(), rt_im.data(), N);
        max_err = 0.0;
        for (int i = 0; i < N; i++) {
            double dr = x_re[i] - rt_re[i];
            double di = x_im[i] - rt_im[i];
            double e = sqrt(dr*dr + di*di);
            if (e > max_err) max_err = e;
        }
        std::printf("  %-6d  %-12s  %8.1f dB   %.4e\n",
                    N, "roundtrip", sqnr_rt, max_err);
    }

    std::printf("# Done.\n");
    return 0;
}
