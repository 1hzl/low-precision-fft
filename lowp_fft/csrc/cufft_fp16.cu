#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// ─── Plan cache ─────────────────────────────────────────────────────
struct CachedPlan {
    cufftHandle plan;
    void* workspace;
    size_t workspace_bytes;
};

static std::unordered_map<std::string, CachedPlan> g_plan_cache;
static std::mutex g_cache_mutex;

static std::string cache_key(int64_t n, int64_t batch, int direction) {
    return std::to_string(n) + "_" + std::to_string(batch) + "_" + std::to_string(direction);
}

static cufftHandle acquire_plan(int64_t n, int64_t batch, int direction) {
    std::string key = cache_key(n, batch, direction);

    {
        std::lock_guard<std::mutex> lk(g_cache_mutex);
        auto it = g_plan_cache.find(key);
        if (it != g_plan_cache.end()) return it->second.plan;
    }

    CachedPlan entry{};
    CUFFT_CHECK(cufftCreate(&entry.plan));

    long long fft_n = n;
    long long fft_batch = batch;
    size_t ws = 0;

    CUFFT_CHECK(cufftXtMakePlanMany(
        entry.plan,
        1,              // rank
        &fft_n,         // n
        nullptr,        // inembed
        1,              // istride
        n,              // idist
        CUDA_C_16F,     // input type
        nullptr,        // onembed
        1,              // ostride
        n,              // odist
        CUDA_C_16F,     // output type
        fft_batch,      // batch
        &ws,            // workSize
        CUDA_C_16F));   // exec type

    if (ws > 0) {
        CUDA_CHECK(cudaMalloc(&entry.workspace, ws));
        CUFFT_CHECK(cufftXtSetWorkArea(entry.plan, &entry.workspace));
    }
    entry.workspace_bytes = ws;

    {
        std::lock_guard<std::mutex> lk(g_cache_mutex);
        auto it = g_plan_cache.find(key);
        if (it != g_plan_cache.end()) {
            // lost the race — discard ours
            if (entry.workspace) cudaFree(entry.workspace);
            cufftDestroy(entry.plan);
            return it->second.plan;
        }
        g_plan_cache[key] = entry;
    }
    return entry.plan;
}

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

    cufftHandle plan = acquire_plan(n, batch, direction);

    // c10::complex<at::Half> is bit-compatible with cuFFT FP16 interleaved
    CUFFT_CHECK(cufftXtExec(plan, (void*)input.data_ptr(), (void*)output.data_ptr(), direction));

    // cuFFT inverse is unnormalised; match torch.fft.ifft "backward" default
    if (direction == CUFFT_INVERSE) {
        output.div_(static_cast<float>(n));
    }

    return output;
}

torch::Tensor fft_fp16(torch::Tensor input) {
    return fft_fp16_impl(input, CUFFT_FORWARD);
}

torch::Tensor ifft_fp16(torch::Tensor input) {
    return fft_fp16_impl(input, CUFFT_INVERSE);
}

// ─── Module cleanup ─────────────────────────────────────────────────
static void cleanup_plans() {
    std::lock_guard<std::mutex> lk(g_cache_mutex);
    for (auto& [key, entry] : g_plan_cache) {
        cufftDestroy(entry.plan);
        if (entry.workspace) cudaFree(entry.workspace);
    }
    g_plan_cache.clear();
}

// ─── pybind11 ───────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fft_fp16", &fft_fp16, "1D forward FFT via cuFFT FP16 (along last dim)");
    m.def("ifft_fp16", &ifft_fp16, "1D inverse FFT via cuFFT FP16 (along last dim)");

    // Register cleanup so plans are freed on module unload
    m.add_object("_cleanup", py::capsule(cleanup_plans));
}
