#include <stdio.h>
#include <cuda_runtime.h>

__global__
void staticReverse(int *a,int n){
    __shared__ int temp[64];
    int index = threadIdx.x;
    int re = n-index-1;
    
    temp[index]=a[index];
    __syncthreads();
    a[index]=temp[re];

}

__global__
void dynamicReverse(int *a,int n){
    extern __shared__ int temp[];
    int index = threadIdx.x;
    int re = n-index-1;
    
    temp[index]=a[index];
    __syncthreads();
    a[index]=temp[re];

}

__global__
void conflictReverse(int *a,int n){
    __shared__ int temp[64*32];
    int index = threadIdx.x;
    int re = n-index-1;
    
    temp[index*32]=a[index];
    __syncthreads();
    a[index]=temp[re*32];

}

__global__
void sharedConflictFree(int *out, int iterations){
    __shared__ int values[1024];
    int tid = threadIdx.x;

    for (int i = tid; i < 1024; i += blockDim.x) {
        values[i] = i;
    }
    __syncthreads();

    volatile int *shared = values;
    int offset = tid & 31;
    int sum = 0;
    for (int i = 0; i < iterations; ++i) {
        sum += shared[offset];
    }

    out[blockIdx.x * blockDim.x + tid] = sum;
}

__global__
void sharedBankConflict(int *out, int iterations){
    __shared__ int values[1024];
    int tid = threadIdx.x;

    for (int i = tid; i < 1024; i += blockDim.x) {
        values[i] = i;
    }
    __syncthreads();

    volatile int *shared = values;
    int offset = (tid & 31) * 32;
    int sum = 0;
    for (int i = 0; i < iterations; ++i) {
        sum += shared[offset];
    }

    out[blockIdx.x * blockDim.x + tid] = sum;
}

int main(){
    const int n = 64;
    int a[n], r[n], d[n];

    for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
    }

    int *d_d;
    cudaMalloc(&d_d, n * sizeof(int)); 

    //静态版本
    cudaMemcpy(d_d,a,n*sizeof(int),cudaMemcpyHostToDevice);
    staticReverse<<<1,64>>>(d_d,n);
    cudaMemcpy(d,d_d,n*sizeof(int),cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

    //动态版本
    cudaMemcpy(d_d,a,n*sizeof(int),cudaMemcpyHostToDevice);
    dynamicReverse<<<1,64,n*sizeof(int)>>>(d_d,n);
    cudaMemcpy(d,d_d,n*sizeof(int),cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

    //conflict版本
    cudaMemcpy(d_d,a,n*sizeof(int),cudaMemcpyHostToDevice);
    conflictReverse<<<1,64,n*sizeof(int)>>>(d_d,n);
    cudaMemcpy(d,d_d,n*sizeof(int),cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

    cudaFree(d_d);

    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sm_count * 6;
    int *d_out;
    cudaMalloc(&d_out, blocks * 256 * sizeof(int));

    sharedConflictFree<<<blocks, 256>>>(d_out, 100000);
    sharedBankConflict<<<blocks, 256>>>(d_out, 100000);

    cudaDeviceSynchronize();
    cudaFree(d_out);
}
