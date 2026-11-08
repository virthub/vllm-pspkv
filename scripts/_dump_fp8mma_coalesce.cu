// Dump the COALESCED (compact) A/B/C layouts: the codomain that mma_unpack's
// recast<> actually packs into registers. value = idx % ValSize, thread = idx / ValSize.
#include <cuda_runtime.h>
#include <cstdio>
#include <cute/atom/mma_traits_sm89.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>

using namespace cute;

int main() {
    using Traits = MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN>;
    auto A = coalesce(Traits::ALayout{});
    auto B = coalesce(Traits::BLayout{});
    auto C = coalesce(Traits::CLayout{});
    printf("A coalesced: shape = "); print(A.shape()); printf("  stride = "); print(A.stride()); printf("\n");
    printf("B coalesced: shape = "); print(B.shape()); printf("  stride = "); print(B.stride()); printf("\n");
    printf("C coalesced: shape = "); print(C.shape()); printf("  stride = "); print(C.stride()); printf("\n");

    // A: (M,K)=(16,32), ValSize=16. print value:thread for each (m,k)
    printf("A (value:thread) M x K (8 rows x 32 cols):\n");
    for (int m = 0; m < 8; m++) {
        printf(" m=%d:", m);
        for (int k = 0; k < 32; k++) { int idx = A(m,k); printf(" %02d:%02d", idx%16, idx/16); }
        printf("\n");
    }
    printf("B (value:thread) N x K (8 rows x 32 cols):\n");
    for (int n = 0; n < 8; n++) {
        printf(" n=%d:", n);
        for (int k = 0; k < 32; k++) { int idx = B(n,k); printf(" %d:%02d", idx%8, idx/8); }
        printf("\n");
    }
    printf("C (value:thread) M x N (16 rows x 8 cols):\n");
    for (int m = 0; m < 16; m++) {
        printf(" m=%2d:", m);
        for (int n = 0; n < 8; n++) { int idx = C(m,n); printf(" %d:%02d", idx%4, idx/4); }
        printf("\n");
    }
    return 0;
}
