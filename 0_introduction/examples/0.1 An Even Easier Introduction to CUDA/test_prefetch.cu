#include <cuda_runtime.h>
#include <iostream>

__global__
void add(float *x,float *y,float *sum, int n){
    int stride = gridDim.x*blockDim.x;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(int i = index;i<n;i+=stride){
        sum[i]=x[i]+y[i];
    }
    
}
/* blockdim:块大小（每个块中线程数量）
gridDim:块的数量
threadIdx：一个块中thread的索引
blockIdx：网格中block的索引

gridDim*blockDim:线程总数
blockIdx*blockDim:block的初始位置
blockIdx*blockDim+threadIdx:block中某个thread的初始位置

网格跨步循环：所有thread（block大小*block数量）处理一大块数据，一步大小为所有thread的数量
*/


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

    cudaMemLocation location{};
    location.type = cudaMemLocationTypeDevice;
    location.id = 0;  // GPU 编号，单卡填 0

    cudaMemPrefetchAsync(x,n*sizeof(float),location,0);
    cudaMemPrefetchAsync(y,n*sizeof(float),location,0);

    add<<<2048, 256>>>(x, y, y, n);

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