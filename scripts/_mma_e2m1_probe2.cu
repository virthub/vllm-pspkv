// Determine e2m1 encoding + MMA semantics precisely using cutlass's own types.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include "cutlass/float_subbyte.h"
using namespace cutlass;

__global__ void mma(float* out, uint32_t av, uint32_t bv) {
    uint32_t a0=av,a1=av,a2=av,a3=av, b0=bv,b1=bv;
    float c0=0,c1=0,c2=0,c3=0;
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    if (threadIdx.x == 0) out[0]=c0,out[1]=c1,out[2]=c2,out[3]=c3;
}

int main() {
    // host-side: encode 1.0f and 2.0f and 0.5f into e2m1
    float_e2m1_t e1(1.0f), e2(2.0f), eh(0.5f);
    printf("e2m1(1.0)=0x%x  e2m1(2.0)=0x%x  e2m1(0.5)=0x%x  e2m1(0.25)=0x%x\n",
           (unsigned)e1.raw(), (unsigned)e2.raw(), (unsigned)eh.raw(),
           (unsigned)float_e2m1_t(0.25f).raw());
    uint32_t one = (unsigned)e1.raw();
    uint32_t all_ones = one*0x11111111u;  // 8 nibbles of e2m1(1.0)
    float* o; cudaMallocManaged(&o, 4);
    mma<<<1,32>>>(o, all_ones, all_ones); cudaDeviceSynchronize();
    printf("all-ones(1.0): c0=%.1f c1=%.1f c2=%.1f c3=%.1f  (32=plain K32)\n", o[0],o[1],o[2],o[3]);
    return 0;
}
