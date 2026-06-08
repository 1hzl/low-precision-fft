#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cufft.h>
#include <cufftXt.h>
#include <unordered_map>
#include <mutex>
#include <string>

// ─── Error helpers ──────────────────────────────────────────────────
static void check_cufft(cufftResult r, const char* file, int line) {
    if (r != CUFFT_SUCCESS) {
        throw std::runtime_error(
            std::string("cuFFT error ") + std::to_string((int)r) +
            " at " + file + ":" + std::to_string(line));
    }
}
#define CUFFT_CHECK(call) check_cufft((call), __FILE__, __LINE__)

static void check_cuda(cudaError_t r, const char* file, int line) {
    if (r != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(r) +
            " at " + file + ":" + std::to_string(line));
    }
}
#define CUDA_CHECK(call) check_cuda((call), __FILE__, __LINE__)

// ─── Plan cache (templated for FP16 / BF16 reuse) ───────────────────
struct CachedPlan {
    cufftHandle plan;
    void* workspace;
    size_t workspace_bytes;
};

static constexpr size_t kMaxCacheEntries = 64;

template<cudaDataType DType>
struct PlanCache {
    static std::unordered_map<std::string, CachedPlan> plans;
    static std::mutex mtx;

    static std::string make_key(int64_t n, int64_t batch, int direction) {
        return std::to_string(n) + "_" + std::to_string(batch) + "_" + std::to_string(direction);
    }

    static cufftHandle acquire(int64_t n, int64_t batch, int direction) {
        std::string key = make_key(n, batch, direction);

        {
            std::lock_guard<std::mutex> lk(mtx);
            auto it = plans.find(key);
            if (it != plans.end()) return it->second.plan;
        }

        CachedPlan entry{};
        CUFFT_CHECK(cufftCreate(&entry.plan));

        long long fft_n = n;
        long long fft_batch = batch;
        size_t ws = 0;

        CUFFT_CHECK(cufftXtMakePlanMany(
            entry.plan, 1, &fft_n, nullptr, 1, n, DType,
            nullptr, 1, n, DType, fft_batch, &ws, DType));

        if (ws > 0) {
            CUDA_CHECK(cudaMalloc(&entry.workspace, ws));
            CUFFT_CHECK(cufftXtSetWorkArea(entry.plan, &entry.workspace));
        }
        entry.workspace_bytes = ws;

        {
            std::lock_guard<std::mutex> lk(mtx);
            auto it = plans.find(key);
            if (it != plans.end()) {
                if (entry.workspace) cudaFree(entry.workspace);
                cufftDestroy(entry.plan);
                return it->second.plan;
            }
            if (plans.size() >= kMaxCacheEntries) {
                for (auto& [k, e] : plans) {
                    cufftDestroy(e.plan);
                    if (e.workspace) cudaFree(e.workspace);
                }
                plans.clear();
            }
            plans[key] = entry;
        }
        return entry.plan;
    }

    static void cleanup() {
        std::lock_guard<std::mutex> lk(mtx);
        for (auto& [key, entry] : plans) {
            cufftDestroy(entry.plan);
            if (entry.workspace) cudaFree(entry.workspace);
        }
        plans.clear();
    }
};

template<cudaDataType DType>
std::unordered_map<std::string, CachedPlan> PlanCache<DType>::plans;
template<cudaDataType DType>
std::mutex PlanCache<DType>::mtx;

// ─── Forward / inverse ──────────────────────────────────────────────
static torch::Tensor fft_fp16_impl(torch::Tensor input, int direction) {
    TORCH_CHECK(input.is_cuda(), "fft_fp16 requires CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "fft_fp16 requires contiguous tensor");
    TORCH_CHECK(input.scalar_type() == torch::kComplexHalf,
                "fft_fp16 requires complex32 (ComplexHalf) input");
    TORCH_CHECK(input.dim() >= 1, "fft_fp16 requires at least 1-dimensional input");

    int64_t n = input.size(-1);
    int64_t batch = input.numel() / n;

    auto output = torch::empty_like(input);

    cufftHandle plan = PlanCache<CUDA_C_16F>::acquire(n, batch, direction);

    // c10::complex<at::Half> is bit-compatible with cuFFT FP16 interleaved
    CUFFT_CHECK(cufftXtExec(plan, (void*)input.data_ptr(), (void*)output.data_ptr(), direction));

    return output;
}

torch::Tensor fft_fp16_forward(torch::Tensor input) {
    return fft_fp16_impl(input, CUFFT_FORWARD);
}

torch::Tensor ifft_fp16_forward(torch::Tensor input) {
    return fft_fp16_impl(input, CUFFT_INVERSE);
}

// ─── BF16 forward / inverse ─────────────────────────────────────────
static torch::Tensor fft_bf16_impl(torch::Tensor input, int direction) {
    TORCH_CHECK(input.is_cuda(), "fft_bf16 requires CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "fft_bf16 requires contiguous tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
                "fft_bf16 requires bfloat16 input");
    TORCH_CHECK(input.dim() >= 2, "fft_bf16 requires at least 2-dimensional input (last dim = real/imag)");
    TORCH_CHECK(input.size(-1) == 2,
                "fft_bf16 requires last dim == 2 (real/imag interleaved), got ",
                input.size(-1));

    int64_t n = input.size(-2);
    int64_t batch = input.numel() / (n * 2);

    auto output = torch::empty_like(input);

    cufftHandle plan = PlanCache<CUDA_C_16BF>::acquire(n, batch, direction);

    CUFFT_CHECK(cufftXtExec(plan, (void*)input.data_ptr(), (void*)output.data_ptr(), direction));

    return output;
}

torch::Tensor fft_bf16_forward(torch::Tensor input) {
    return fft_bf16_impl(input, CUFFT_FORWARD);
}

torch::Tensor ifft_bf16_forward(torch::Tensor input) {
    return fft_bf16_impl(input, CUFFT_INVERSE);
}

// ─── Module cleanup ─────────────────────────────────────────────────
static void cleanup_plans() {
    PlanCache<CUDA_C_16F>::cleanup();
    PlanCache<CUDA_C_16BF>::cleanup();
}

// ─── pybind11 ───────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fft_fp16_forward", &fft_fp16_forward, "1D forward FFT via cuFFT FP16 (along last dim)");
    m.def("ifft_fp16_forward", &ifft_fp16_forward, "1D inverse FFT via cuFFT FP16 (along last dim)");

    m.def("fft_bf16_forward", &fft_bf16_forward, "1D forward FFT via cuFFT BF16 (along dim=-2, last dim=real/imag)");
    m.def("ifft_bf16_forward", &ifft_bf16_forward, "1D inverse FFT via cuFFT BF16 (along dim=-2, last dim=real/imag)");

    // Register cleanup so plans are freed on module unload
    m.add_object("_cleanup", py::capsule(cleanup_plans));
}
