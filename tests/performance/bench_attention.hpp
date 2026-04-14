#include <torch/torch.h>

#include "bench.hpp"
#include "math.hpp"
#include "ops.hpp"

using namespace attn;

namespace bench {

void run_attention_benchmarks() {
    bench::Context ctx;
    bench::Runner runner("Scaled Dot-Product Attention", ctx);

    bench::SharedTensor Q_shared(ctx.batch_size, ctx.seq_len_q, ctx.d_k);
    bench::SharedTensor K_shared(ctx.batch_size, ctx.seq_len_k, ctx.d_k);
    bench::SharedTensor V_shared(ctx.batch_size, ctx.seq_len_k, ctx.d_v);

    Tensor Q = Q_shared.as_tensor();
    Tensor K = K_shared.as_tensor();
    Tensor V = V_shared.as_tensor();

    runner.run("My attention naive", [&]() { ops::attention(Q, K, V, attn::math::MatMulType::NAIVE); });
    runner.run("My attention cf", [&]() { ops::attention(Q, K, V, attn::math::MatMulType::CACHE_OPTIMIZED); });
    runner.run("My attention simd", [&]() { ops::attention(Q, K, V, attn::math::MatMulType::SIMD); });

    auto t_Q = Q_shared.as_torch();
    auto t_K = K_shared.as_torch();
    auto t_V = V_shared.as_torch();

    runner.run("LibTorch", [&]() { torch::scaled_dot_product_attention(t_Q, t_K, t_V); });

    runner.print();
}

} // namespace bench
