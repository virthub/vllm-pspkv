// Probe which 4-bit MMA instructions assemble on SM120.
#include <cuda_runtime.h>
#include <cstdio>
#include <stdint.h>

// Try block-scaled fp4 MMA variants; a non-assembling one errors out.
__global__ void probe_kind_mxf4(int* out) {
    uint32_t a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    float c0=0,c1=0,c2=0,c3=0;
#if defined(TRY_M16N8K64_MXF4)
    asm volatile("mma.sync.aligned.kind::mxf4.m16n8k64.row.col.f32.e2m1.e2m1.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#elif defined(TRY_M16N16K32_MXF4)
    asm volatile("mma.sync.aligned.kind::mxf4.m16n16k32.row.col.f32.e2m1.e2m1.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#endif
    if (threadIdx.x == 0) out[0] = (int)c0;
}
int main(){ int* o; cudaMallocManaged(&o,4); probe_kind_mxf4<<<1,32>>>(o); cudaDeviceSynchronize();
  printf("assembled+ran ok (result %d)\n", *o); return 0; }
