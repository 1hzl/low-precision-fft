/**
 * bfp_fft.h — Block Floating-Point FFT CUDA API (Sprint 3.3)
 *
 * Implements per-stage BFP Radix-2 DIT FFT using __nv_fp8_e4m3 hardware type.
 * Each FFT stage shares one integer exponent; butterflies run in float32;
 * output is requantized to FP8 mantissas with a new shared exponent.
 */

#ifndef BFP_FFT_H
#define BFP_FFT_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #ifdef BFP_FFT_EXPORT
    #define BFP_API __declspec(dllexport)
  #else
    #define BFP_API __declspec(dllimport)
  #endif
#else
  #define BFP_API
#endif

/**
 * BFP forward FFT.
 *
 * @param x_real      Input real parts (N elements, natural order)
 * @param x_imag      Input imag parts (N elements, natural order)
 * @param X_real      Output real parts (N elements, natural order)
 * @param X_imag      Output imag parts (N elements, natural order)
 * @param N           FFT size (must be power of 2)
 * @param stages_exp  Output: shared exponents (log2(N)+1 elements)
 * @return 0 on success, -1 on error
 */
BFP_API int bfp_fft_forward(
    const float* x_real, const float* x_imag,
    float* X_real, float* X_imag,
    int N,
    int* stages_exp
);

/**
 * BFP inverse FFT.
 *
 * @param x_real      Input real parts (N elements, natural order)
 * @param x_imag      Input imag parts (N elements, natural order)
 * @param X_real      Output real parts (N elements, natural order)
 * @param X_imag      Output imag parts (N elements, natural order)
 * @param N           FFT size (must be power of 2)
 * @param stages_exp  Output: shared exponents (log2(N)+1 elements)
 * @return 0 on success, -1 on error
 */
BFP_API int bfp_fft_inverse(
    const float* x_real, const float* x_imag,
    float* X_real, float* X_imag,
    int N,
    int* stages_exp
);

/**
 * Compute SQNR between reference and test signals.
 *
 * @param ref_real  Reference real parts
 * @param ref_imag  Reference imag parts
 * @param tst_real  Test real parts
 * @param tst_imag  Test imag parts
 * @param N         Number of elements
 * @return SQNR in dB
 */
BFP_API double bfp_compute_sqnr(
    const float* ref_real, const float* ref_imag,
    const float* tst_real, const float* tst_imag,
    int N
);

#ifdef __cplusplus
}
#endif

#endif /* BFP_FFT_H */
