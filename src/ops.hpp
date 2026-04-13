#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "math.hpp"
#include "tensor.hpp"
#include "validation.hpp"

namespace attn::ops {

namespace details {

inline void softmax_row(Tensor& input, std::size_t batch_idx, std::size_t row_idx) {
    std::size_t cols = input.get_cols();

    float* row_start = &input(batch_idx, row_idx, 0);
    float* row_end = row_start + cols;

    float max_val = *std::max_element(row_start, row_end);

    float sum_exp = 0.0f;
    for (std::size_t j = 0; j < cols; ++j) {
        float e_j = std::exp(row_start[j] - max_val);
        row_start[j] = e_j;
        sum_exp += e_j;
    }

    std::transform(row_start, row_end, row_start, [sum_exp](float val) { return val / sum_exp; });
}

inline void softmax_batch(Tensor& input, std::size_t batch_idx) {
    for (std::size_t i = 0; i < input.get_rows(); ++i) {
        softmax_row(input, batch_idx, i);
    }
}

} // namespace details

inline void softmax(Tensor& input) {
    for (std::size_t b = 0; b < input.get_batch(); ++b) {
        details::softmax_batch(input, b);
    }
}

inline Tensor attention(const Tensor& queries, const Tensor& keys, const Tensor& values) {
    details::validate_attention_dimensions(queries, keys, values);

    Tensor scores = math::multiply_tr(queries, keys);

    float d_k = static_cast<float>(queries.get_cols());
    float scale_factor = 1 / std::sqrt(d_k);
    math::scale(scores, scale_factor);

    softmax(scores);

    Tensor values_tr = math::transpose(values);

    return math::multiply_tr(scores, values_tr);
}

} // namespace attn::ops
