#pragma once

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "common.hpp"

namespace bench {

struct Context {
    int64_t batch_size = 4;
    int64_t seq_len_q = 1024;
    int64_t seq_len_k = 1024;
    int64_t d_k = 64;
    int64_t d_v = 128;

    std::size_t warmup = 5;
    std::size_t repeats = 20;
};

struct Result {
    std::string name;
    double time_ms;
};

class Runner {
  public:
    explicit Runner(const std::string& title, Context ctx = Context{}) : title_(title), ctx_(ctx) {}

    template <typename F, typename... Args>
    void run(const std::string& name, F&& func, Args&&... args) {
        double time_ms = profile_function(ctx_.repeats, ctx_.warmup, std::forward<F>(func),
                                          std::forward<Args>(args)...);
        results_.push_back({name, time_ms});
    }

    void print() const {
        // clang-format off
        std::cout << "\n=== BENCHMARK: " << title_ << " ===\n";

        std::cout << "Dimensions: B=" << ctx_.batch_size 
                           << " | N=" << ctx_.seq_len_q  
                           << " | M=" << ctx_.seq_len_k 
                           << " | d_k=" << ctx_.d_k
                           << " | d_v=" << ctx_.d_v
                           << "\n";

        std::cout << std::left << std::setw(25) << "Implementation" 
                  << std::right << std::setw(15) << "Time (ms)" 
                  << "\n";

        std::cout << std::string(55, '-') << std::endl;
        // clang-format on

        for (const auto& res : results_) {
            std::cout << std::left << std::setw(25) << res.name << std::right << std::setw(15)
                      << std::fixed << std::setprecision(4) << res.time_ms
                      << std::endl;
        }
        std::cout << std::string(55, '=') << std::endl;
    }

  private:
    std::string title_;
    Context ctx_;
    std::vector<Result> results_;
};

class SharedTensor {
  private:
    inline static const torch::TensorOptions options =
        torch::TensorOptions().dtype(torch::kFloat32);

  public:
    SharedTensor(int64_t batch, int64_t rows, int64_t cols, float min_val = -1.0f,
                 float max_val = 1.0f)
        : b_(batch), r_(rows), c_(cols) {

        data_.resize(b_ * r_ * c_);
        RandFill(data_, min_val, max_val);
    }

    attn::Tensor as_tensor() { return attn::Tensor::make_view(data_.data(), b_, r_, c_); }

    torch::Tensor as_torch() { return torch::from_blob(data_.data(), {b_, r_, c_}, options); }

  private:
    std::vector<float> data_;
    int64_t b_, r_, c_;
};

} // namespace bench
