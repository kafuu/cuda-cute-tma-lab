#include <cuda_runtime.h>
#include <iostream>

__global__
void add1(float *x,float *y,float *sum, int n){
    int thread = blockDim.x;
    int index = threadIdx.x;
    int blocksize = n/thread;
    for(int i = index*blocksize;i<(index+1)*blocksize ;i++){
        sum[i]=x[i]+y[i];
    }
    
}

__global__
void add(float *x,float *y,float *sum, int n){
    int stride = blockDim.x;
    int index = threadIdx.x;
    for(int i = index;i<n ;i+=stride){
        sum[i]=x[i]+y[i];
    }
    
}
//这种方法即使块大小不为整数也正确


int main(){
    float *x, *y;
    int n;
    n = 1<<20;
    cudaMallocManaged(&x,n*sizeof(float));
    cudaMallocManaged(&y,n*sizeof(float));

    for(int i  = 0;i < n;i++){
        x[i]=1.0f;
        y[i]=2.0f;
    }


/*     cudaMemLocation location{};
    location.type = cudaMemLocationTypeDevice;
    location.id = 0;  // GPU 编号，单卡填 0

    cudaMemPrefetchAsync(x,n*sizeof(float),location,0);
    cudaMemPrefetchAsync(y,n*sizeof(float),location,0);
 */

    add<<<1,256>>>(x,y,y,n);

    cudaDeviceSynchronize();

    float maxerror = 0.0f;
    for(int i = 0;i < n; i++){
        maxerror = fmax(maxerror,fabs(y[i]-3.0f));
    }

    std::cout<< "maxerror:" << maxerror << std::endl;
    
    cudaFree(x);
    cudaFree(y);

    return 0;
}