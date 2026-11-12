// Probe the SM120 4-bit (FP4 e2m1) MMA: semantics + fragment layout.
#include <cuda_runtime.h>
#include <cstdio>
#include <stdint.h>

// e2m1 encoding: bit3=sign, bit2:1=exp(2b,bias1), bit0=mantissa. 1.0 = 0b0010 = 0x2.
// All 8 nibbles = 1.0 -> 0x22222222.
__global__ void probe_allones(float* out) {
    uint32_t a0=0x22222222u,a1=0x22222222u,a2=0x22222222u,a3=0x22222222u;
    uint32_t b0=0x22222222u,b1=0x22222222u;
    float c0=0,c1=0,c2=0,c3=0;
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    if (threadIdx.x < 32) out[threadIdx.x*4+0]=c0, out[threadIdx.x*4+1]=c1, out[threadIdx.x*4+2]=c2, out[threadIdx.x*4+3]=c3;
}

// Probe layout: set a single nibble in a0 (lane 0), see which (m,n) lights up.
__global__ void probe_single(float* out, int reg, int nib) {
    uint32_t a0=0,a1=0,a2=0,a3=0;
    uint32_t b0=0x22222222u,b1=0x22222222u;   // B all 1.0
    uint32_t* ar[4] = {&a0,&a1,&a2,&a3};
    *ar[reg] = 0x2u << (4*nib);               // single e2m1=1.0 at nibble `nib` of reg `reg`
    float c0=0,c1=0,c2=0,c3=0;
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    if (threadIdx.x == 0) out[0]=c0,out[1]=c1,out[2]=c2,out[3]=c3;
}

int main(){
  float* o; cudaMallocManaged(&o, 128*4);
  probe_allones<<<1,32>>>(o); cudaDeviceSynchronize();
  // find max |value| and print unique values
  float mx=0; for(int i=0;i<128;i++){float v=o[i]; if(v>mx)mx=v;}
  printf("all-ones: max output = %.1f  (32=plain K32, 16=blockscaled x2, 8=blockscaled x4)\n", mx);
  // layout probe: for each (reg 0..3, nib 0..7), check lane0 output
  printf("lane0 single-nibble A -> (c0,c1,c2,c3):\n");
  for(int r=0;r<4;r++){ printf("  reg%d:", r); for(int n=0;n<8;n++){ probe_single<<<1,32>>>(o, r, n); cudaDeviceSynchronize();
      printf(" [%d]=%.0f", n, o[0]); } printf("\n"); }
  return 0;
}
