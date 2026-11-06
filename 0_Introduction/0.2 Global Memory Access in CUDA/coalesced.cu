
__global__
void coalescedAccess(float *input,int n){
    int index = blockDim.x*blockIdx.x+threadIdx.x;
    if(index<n){
        input[index]=input[index]*2.0f;
    }

}

__global__
void uncoalescedAccess(float *input,int n){
    int index = blockDim.x*blockIdx.x+threadIdx.x;
    index = index*32%n;
    if(index<n){
        input[index]=input[index]*2.0f;
    }
}
int main(){
    float *a;
    int n = 1<<22;

    cudaMallocManaged(&a,n*sizeof(float));

    for(int i=0;i<n;i++){
        a[i]=1;
    }

    cudaMemLocation location{};
    location.type = cudaMemLocationTypeDevice;
    location.id = 0;  // GPU 编号，单卡填 0

    cudaMemPrefetchAsync(a,n*sizeof(float),location,0);

    coalescedAccess<<<256,256>>>(a,n);
    uncoalescedAccess<<<256,256>>>(a,n);

    cudaDeviceSynchronize();

    cudaFree(a);

    return 0;
}