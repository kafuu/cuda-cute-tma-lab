#include <cuda_runtime.h>
#include <iostream>

__global__
void add(float *x,float *y,float *sum, int n){
    for(int i = 0;i<n ;i++){
        sum[i]=x[i]+y[i];
    }
    
}

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

    add<<<1,1>>>(x,y,y,n);

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