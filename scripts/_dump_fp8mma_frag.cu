// Host-side dump of the thread-fragment layouts (thrfrg_A/B/C).
#include <cuda_runtime.h>
#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm89.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;
using fp8 = cutlass::float_e4m3_t;

int main() {
    using MMA_T = SM89_16x8x32_F32E4M3E4M3F32_TN;
    auto mma = make_tiled_mma(MMA_T{});
    auto a = make_layout(make_shape(Int<16>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto b = make_layout(make_shape(Int<8>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto c = make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
    auto tA = mma.thrfrg_A(a);
    auto tB = mma.thrfrg_B(b);
    auto tC = mma.thrfrg_C(c);
    printf("thrfrg_A shape: "); print(tA); printf("\n");
    printf("thrfrg_B shape: "); print(tB); printf("\n");
    printf("thrfrg_C shape: "); print(tC); printf("\n");
    return 0;
}
