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
    if (threadIdx.x==0) out[0]=c0,out[1]=c1,out[2]=c2,out[3]=c3;
}
int main(){
  float* o; cudaMallocManaged(&o,4);
  auto enc=[&](float x){ return (unsigned)float_e2m1_t(x).raw(); };
  // values: 1.0, 2.0, 0.5, and a mix
  uint32_t one=enc(1.0f)*0x11111111u;
  mma<<<1,32>>>(o, one, one); cudaDeviceSynchronize();
  printf("A=B=1.0 : c0=%.1f (expect 32 if plain K32)\n", o[0]);
  uint32_t two=enc(2.0f)*0x11111111u;
  mma<<<1,32>>>(o, two, two); cudaDeviceSynchronize();
  printf("A=B=2.0 : c0=%.1f\n", o[0]);
  uint32_t half=enc(0.5f)*0x11111111u;
  mma<<<1,32>>>(o, half, half); cudaDeviceSynchronize();
  printf("A=B=0.5 : c0=%.1f\n", o[0]);
  // A=1.0, B=2.0 -> mixed
  mma<<<1,32>>>(o, one, two); cudaDeviceSynchronize();
  printf("A=1.0,B=2.0 : c0=%.1f\n", o[0]);
  return 0;
}
