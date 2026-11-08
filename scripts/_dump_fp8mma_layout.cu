// Dump the CUTLASS SM89_16x8x32 fp8 MMA fragment layouts as raw codomain
// indices so the (value, thread) mapping can be decoded directly.
#include <cuda_runtime.h>
#include <cstdio>
#include <cute/atom/mma_traits_sm89.hpp>
#include <cute/layout.hpp>

using namespace cute;

int main() {
    using Traits = MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN>;
    auto A = Traits::ALayout{};
    auto B = Traits::BLayout{};
    auto C = Traits::CLayout{};
    // A: (M,K)=(16,32), ValSize=16. B: (N,K)=(8,32), ValSize=8. C: (M,N)=(16,8), ValSize=4.
    printf("A raw idx (M=16 x K=32):\n");
    for (int m = 0; m < 16; m++) {
        printf(" m=%2d:", m);
        for (int k = 0; k < 32; k++) printf(" %3d", A(m, k));
        printf("\n");
    }
    printf("B raw idx (N=8 x K=32):\n");
    for (int n = 0; n < 8; n++) {
        printf(" n=%d:", n);
        for (int k = 0; k < 32; k++) printf(" %3d", B(n, k));
        printf("\n");
    }
    printf("C raw idx (M=16 x N=8):\n");
    for (int m = 0; m < 16; m++) {
        printf(" m=%2d:", m);
        for (int n = 0; n < 8; n++) printf(" %3d", C(m, n));
        printf("\n");
    }
    return 0;
}
