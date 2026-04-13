#pragma once

#include <cstddef>
#include <iterator>
#include <stdexcept>

namespace attn::details {

constexpr inline const char* kErrInvalidDimensions =
    "Invalid tensor dimensions: cannot mix zero and non-zero dimensions.";

constexpr inline const char* kErrInputDataAndDimensionsMismatch =
    "Input data size does not match tensor dimensions.";

inline void validate_dimensions(std::size_t batches, std::size_t rows, std::size_t cols) {
#ifndef NDEBUG
    bool has_zero = (batches == 0 || rows == 0 || cols == 0);
    bool all_zero = (batches == 0 && rows == 0 && cols == 0);

    if (has_zero && !all_zero) { throw std::invalid_argument(kErrInvalidDimensions); }
#endif
}

template <typename T, std::forward_iterator FwdIter>
void validate_data_size(const T& tensor, FwdIter begin, FwdIter end) {
#ifndef NDEBUG
    if (tensor.get_n_elems() != std::distance(begin, end)) {
        throw std::invalid_argument(kErrInputDataAndDimensionsMismatch);
    }
#endif
}

} // namespace attn::details


namespace attn::math::details {

constexpr inline const char* kErrTransposeMismatch =
    "Transpose dimension mismatch: result must be (cols x rows) of input.";
constexpr inline const char* kErrTransposeBatchMismatch =
    "Transpose batch mismatch: input and result must have the same number of batches.";

constexpr inline const char* kErrMultiplyMismatch =
    "Multiply dimension mismatch: lhs cols must equal rhs rows.";
constexpr inline const char* kErrMultiplyTrMismatch =
    "Multiply (transposed) dimension mismatch: lhs cols must equal rhs_tr cols.";
constexpr inline const char* kErrMultiplyBatchMismatch =
    "Multiply batch mismatch: lhs, rhs, and result must have the same number of batches.";
constexpr inline const char* kErrMultiplyResultMismatch =
    "Multiply dimension mismatch: result dimensions are incorrect.";

template <typename T>
void validate_transpose_dimensions(const T& input, const T& result) {
#ifndef NDEBUG
    if (input.get_batch() != result.get_batch()) {
        throw std::invalid_argument(kErrTransposeBatchMismatch);
    }

    if (result.get_rows() != input.get_cols() || result.get_cols() != input.get_rows()) {
        throw std::invalid_argument(kErrTransposeMismatch);
    }
#endif
}

template <typename T>
void validate_multiply_dimensions(const T& lhs, const T& rhs, const T& result) {
#ifndef NDEBUG
    if (lhs.get_batch() != rhs.get_batch() || lhs.get_batch() != result.get_batch()) {
        throw std::invalid_argument(kErrMultiplyBatchMismatch);
    }

    if (lhs.get_cols() != rhs.get_rows()) { throw std::invalid_argument(kErrMultiplyMismatch); }

    if (result.get_rows() != lhs.get_rows() || result.get_cols() != rhs.get_cols()) {
        throw std::invalid_argument(kErrMultiplyResultMismatch);
    }
#endif
}

template <typename T>
void validate_multiply_tr_dimensions(const T& lhs, const T& rhs_tr, const T& result) {
#ifndef NDEBUG
    if (lhs.get_batch() != rhs_tr.get_batch() || lhs.get_batch() != result.get_batch()) {
        throw std::invalid_argument(kErrMultiplyBatchMismatch);
    }

    if (lhs.get_cols() != rhs_tr.get_cols()) {
        throw std::invalid_argument(kErrMultiplyTrMismatch);
    }

    if (result.get_rows() != lhs.get_rows() || result.get_cols() != rhs_tr.get_rows()) {
        throw std::invalid_argument(kErrMultiplyResultMismatch);
    }
#endif
}

} // namespace attn::math::details

namespace attn::ops::details {

constexpr inline const char* kErrAttentionBatchMismatch =
    "Attention batch mismatch: queries, keys, and values must have the same number of batches.";
constexpr inline const char* kErrAttentionHeadDimMismatch =
    "Attention dimension mismatch: queries and keys must have the same column dimension (d_k).";
constexpr inline const char* kErrAttentionSeqLenMismatch =
    "Attention dimension mismatch: keys and values must have the same row dimension.";

template <typename T>
inline void validate_attention_dimensions(const T& queries, const T& keys, const T& values) {
#ifndef NDEBUG
    if (queries.get_batch() != keys.get_batch() || queries.get_batch() != values.get_batch()) {
        throw std::invalid_argument(kErrAttentionBatchMismatch);
    }

    if (queries.get_cols() != keys.get_cols()) {
        throw std::invalid_argument(kErrAttentionHeadDimMismatch);
    }

    if (keys.get_rows() != values.get_rows()) {
        throw std::invalid_argument(kErrAttentionSeqLenMismatch);
    }
#endif
}

} // namespace attn::ops::details
