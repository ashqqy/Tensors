#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>

#include "validation.hpp"

namespace attn {

namespace details {

constexpr auto no_delete = [](float*) noexcept {};

} // namespace details

class Tensor {
  public:
    Tensor() = default;

    Tensor(std::size_t n_batches, std::size_t n_rows, std::size_t n_cols)
        : n_batches_(n_batches), n_rows_(n_rows), n_cols_(n_cols),
          data_(std::make_shared<float[]>(n_batches * n_rows * n_cols)) {
        details::validate_dimensions(n_batches, n_rows, n_cols);
    }

    template <std::forward_iterator InputIter>
    Tensor(std::size_t n_batches, std::size_t n_rows, std::size_t n_cols, InputIter begin,
           InputIter end)
        : n_batches_(n_batches), n_rows_(n_rows), n_cols_(n_cols),
          data_(std::make_shared<float[]>(n_batches * n_rows * n_cols)) {
        details::validate_dimensions(n_batches, n_rows, n_cols);
        details::validate_data_size(*this, begin, end);

        std::copy(begin, end, data());
    }

    static Tensor make_view(float* ptr, std::size_t n_batches, std::size_t n_rows,
                            std::size_t n_cols) {
        Tensor tensor;
        tensor.n_batches_ = n_batches;
        tensor.n_rows_ = n_rows;
        tensor.n_cols_ = n_cols;
        tensor.data_ = std::shared_ptr<float[]>(ptr, details::no_delete);
        return tensor;
    }

  public:
    float& operator[](std::size_t global_idx) noexcept { return data_[global_idx]; }
    const float& operator[](std::size_t global_idx) const noexcept { return data_[global_idx]; }

    float& operator()(std::size_t b, std::size_t i, std::size_t j) noexcept {
        return data_[b * get_rows() * get_cols() + i * get_cols() + j];
    }

    const float& operator()(std::size_t b, std::size_t i, std::size_t j) const noexcept {
        return data_[b * get_rows() * get_cols() + i * get_cols() + j];
    }

    float* data() noexcept { return data_.get(); }
    const float* data() const noexcept { return data_.get(); }

    std::size_t get_batch() const noexcept { return n_batches_; }
    std::size_t get_rows() const noexcept { return n_rows_; }
    std::size_t get_cols() const noexcept { return n_cols_; }
    std::size_t get_n_elems() const noexcept { return get_batch() * get_rows() * get_cols(); }

  public:
    Tensor clone() const {
        return Tensor(n_batches_, n_rows_, n_cols_, data(), data() + get_n_elems());
    }

  private:
    std::size_t n_batches_ = 0;
    std::size_t n_rows_ = 0;
    std::size_t n_cols_ = 0;

    std::shared_ptr<float[]> data_;
};

} // namespace attn
