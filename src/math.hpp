#pragma once

#include <algorithm>
#include <cstddef>
#include <immintrin.h>
#include <numeric>

#include "tensor.hpp"
#include "validation.hpp"

namespace attn::math {

enum class MatMulType { NAIVE, CACHE_OPTIMIZED, SIMD };

namespace details {

inline void transpose_matrix(const Tensor& input, Tensor& result, std::size_t batch_idx) noexcept {
    std::size_t input_offset = batch_idx * input.get_cols() * input.get_rows();
    std::size_t result_offset = batch_idx * result.get_cols() * result.get_rows();

    for (std::size_t i = 0; i < input.get_rows(); ++i) {
        std::size_t input_idx = input_offset + i * input.get_cols();
        std::size_t result_idx = result_offset + i;
        for (std::size_t j = 0; j < input.get_cols(); ++j) {
            result[result_idx] = input[input_idx];
            input_idx += 1;
            result_idx += result.get_cols();
        }
    }
}

inline void multiply_matrix_naive(const Tensor& lhs, const Tensor& rhs, Tensor& result,
                                  std::size_t batch_idx) noexcept {
    for (std::size_t i = 0; i < lhs.get_rows(); ++i) {
        for (std::size_t j = 0; j < rhs.get_cols(); ++j) {
            float sum = 0.0f;

            for (std::size_t k = 0; k < lhs.get_cols(); ++k) {
                sum += lhs(batch_idx, i, k) * rhs(batch_idx, k, j);
            }

            result(batch_idx, i, j) = sum;
        }
    }
}

inline void multiply_matrix_tr(const Tensor& lhs, const Tensor& rhs_tr, Tensor& result,
                               std::size_t batch_idx) {
    for (std::size_t i = 0; i < lhs.get_rows(); ++i) {
        const float* lhs_row_start = &lhs(batch_idx, i, 0);
        const float* lhs_row_end = lhs_row_start + lhs.get_cols();

        for (std::size_t j = 0; j < rhs_tr.get_rows(); ++j) {
            const float* rhs_col_start = &rhs_tr(batch_idx, j, 0);

            result(batch_idx, i, j) =
                std::inner_product(lhs_row_start, lhs_row_end, rhs_col_start, 0.0f);
        }
    }
}

inline void multiply_matrix_cf(const Tensor& lhs, const Tensor& rhs, Tensor& result,
                               std::size_t batch_idx) noexcept {
    for (std::size_t i = 0; i < lhs.get_rows(); ++i) {
        float* res_row = &result(batch_idx, i, 0);

        for (std::size_t k = 0; k < lhs.get_cols(); ++k) {
            float lhs_val = lhs(batch_idx, i, k);
            const float* rhs_row = &rhs(batch_idx, k, 0);

            for (std::size_t j = 0; j < rhs.get_cols(); ++j) {
                res_row[j] += lhs_val * rhs_row[j];
            }
        }
    }
}

inline void multiply_matrix_simd(const Tensor& lhs, const Tensor& rhs, Tensor& result,
                                 std::size_t batch_idx) {

    std::size_t K = lhs.get_cols();
    std::size_t N = rhs.get_cols();

    const float* lhs_base = &lhs(batch_idx, 0, 0);
    const float* rhs_base = &rhs(batch_idx, 0, 0);
    float* res_base = &result(batch_idx, 0, 0);

    for (std::size_t i = 0; i < lhs.get_rows(); ++i) {
        float* res_row = res_base + i * N;
        const float* lhs_row = lhs_base + i * K;

        for (std::size_t k = 0; k < K; ++k) {
            float lhs_val = lhs_row[k];
            const float* rhs_row = rhs_base + k * N;

            __m256 v_lhs = _mm256_set1_ps(lhs_val);

            std::size_t j = 0;
            for (; j + 7 < N; j += 8) {
                __m256 v_rhs = _mm256_loadu_ps(rhs_row + j);
                __m256 v_res = _mm256_loadu_ps(res_row + j);

                v_res = _mm256_fmadd_ps(v_lhs, v_rhs, v_res);

                _mm256_storeu_ps(res_row + j, v_res);
            }

            for (; j < N; ++j) {
                res_row[j] += lhs_val * rhs_row[j];
            }
        }
    }
}

} // namespace details

inline void transpose(const Tensor& input, Tensor& result) {
    details::validate_transpose_dimensions(input, result);

    for (std::size_t b = 0; b < input.get_batch(); ++b) {
        details::transpose_matrix(input, result, b);
    }
}

inline Tensor transpose(const Tensor& input) {
    Tensor result(input.get_batch(), input.get_cols(), input.get_rows());
    transpose(input, result);
    return result;
}

inline void scale(Tensor& tensor, float factor) {
    float* data_start = tensor.data();
    float* data_end = data_start + tensor.get_n_elems();

    std::transform(data_start, data_end, data_start, [factor](float val) { return val * factor; });
}

inline void multiply_naive(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
    details::validate_multiply_dimensions(lhs, rhs, result);

    for (std::size_t b = 0; b < lhs.get_batch(); ++b) {
        details::multiply_matrix_naive(lhs, rhs, result, b);
    }
}

inline Tensor multiply_naive(const Tensor& lhs, const Tensor& rhs) {
    Tensor result(lhs.get_batch(), lhs.get_rows(), rhs.get_cols());
    multiply_naive(lhs, rhs, result);
    return result;
}

inline void multiply_tr(const Tensor& lhs, const Tensor& rhs_tr, Tensor& result) {
    details::validate_multiply_tr_dimensions(lhs, rhs_tr, result);

    for (std::size_t b = 0; b < lhs.get_batch(); ++b) {
        details::multiply_matrix_tr(lhs, rhs_tr, result, b);
    }
}

inline Tensor multiply_tr(const Tensor& lhs, const Tensor& rhs_tr) {
    Tensor result(lhs.get_batch(), lhs.get_rows(), rhs_tr.get_rows());
    multiply_tr(lhs, rhs_tr, result);
    return result;
}

inline void multiply_cf(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
    details::validate_multiply_dimensions(lhs, rhs, result);

    for (std::size_t b = 0; b < lhs.get_batch(); ++b) {
        details::multiply_matrix_cf(lhs, rhs, result, b);
    }
}

inline Tensor multiply_cf(const Tensor& lhs, const Tensor& rhs) {
    Tensor result(lhs.get_batch(), lhs.get_rows(), rhs.get_cols());
    multiply_cf(lhs, rhs, result);
    return result;
}

inline void multiply_simd(const Tensor& lhs, const Tensor& rhs, Tensor& result) {
    details::validate_multiply_dimensions(lhs, rhs, result);

    for (std::size_t b = 0; b < lhs.get_batch(); ++b) {
        details::multiply_matrix_simd(lhs, rhs, result, b);
    }
}

inline Tensor multiply_simd(const Tensor& lhs, const Tensor& rhs) {
    Tensor result(lhs.get_batch(), lhs.get_rows(), rhs.get_cols());
    multiply_simd(lhs, rhs, result);
    return result;
}

inline void multiply(const Tensor& lhs, const Tensor& rhs, Tensor& result, MatMulType type) {
    details::validate_multiply_dimensions(lhs, rhs, result);

    switch (type) {
    case MatMulType::NAIVE:
        multiply_naive(lhs, rhs, result);
        break;
    case MatMulType::CACHE_OPTIMIZED:
        multiply_cf(lhs, rhs, result);
        break;
    case MatMulType::SIMD:
        multiply_simd(lhs, rhs, result);
        break;
    }
}

inline Tensor multiply(const Tensor& lhs, const Tensor& rhs, MatMulType type) {
    Tensor result(lhs.get_batch(), lhs.get_rows(), rhs.get_cols());
    multiply(lhs, rhs, result, type);
    return result;
}

} // namespace attn::math
