# CUDA Learning
## 0.1 An Even Easier Introduction to CUDA
任务：两个1M大小的数组相加
CPU实现：test.cpp
GPU实现：test.cu
使用cudaMallocManaged(*x)为x分配空间
使用cudaFree(*x)释放
使用add<<<para>>>(*x, *y)调用cuda kernel
kernel的定义方式：在function前面加上__global__
同步：cpu需要等待GPU完成任务后再输出，使用cudaDeviceSynchronize()

典型的GPU编程范式：分配shared mem，调用kernel，sync，释放shared mem

### profiling
使用nsys
nsys profile -t cuda --stats=true ./add_cuda

使用nsys_seasy
nsys_easy [-t trace] [-s sample] [-c ctxsw] [-o output] [-r report] command
nsys_easy ./test

![alt text](image-3.png)

### 使用Thread
<<<1,1>>>中的**后一个1**代表一个线程块中的线程数,使用的线程块大小是 32 的倍数；选择 256 个线程是一个合理的大小。
add<<<1, 256>>>(N, x, y);
但是这样修改，只会让每一个线程运行一次
threadIdx.x 包含当前线程在其线程块内的索引，
blockDim.x 包含该线程块中的线程数，需要在kernel内用这些参数
![alt text](image-2.png)
   
### 使用线程块
每个线程块都在单个SM上运行，每个SM最多支持1024个thread
<<<1,256>>>的**第一个参数**代表线程块的数量，组成了所谓的grid（网格）
![alt text](image.png)
gridDim.x为网格中的块数（第一个参数）
blockIdx.x为线程块在网格上的索引

网格跨步循环（grid-stride loop）将 stride 设置为网格中的线程总数（ blockDim.x * gridDim.x ）
blockdim:块大小（每个块中线程数量）
gridDim:块的数量
threadIdx：一个块中thread的索引
blockIdx：网格中block的索引

gridDim*blockDim:线程总数
blockIdx*blockDim:block的初始位置
blockIdx*blockDim+threadIdx:block中某个thread的初始位置

网格跨步循环：所有thread（block大小*block数量）处理一大块数据，一步大小为所有thread的数量

![alt text](image-1.png)
可以看到加速没有想象中的大
### 关于多维线程块
使用多维线程块和网格只是为了方便，不会影响性能。块中的线程会以可预测的方式线性化：第一个索引 x 变化最快，其次是 y ，然后是 z 。这意味着，在线程索引的线性化中， threadIdx.x 的连续值表示连续的线程， threadIdx.y 的步长为 blockDim.x ，而 threadIdx.z 的步长为 blockDim.x * blockDim.y 。这会影响线程如何分配到 warp，详见硬件多线程。
![alt text](image-15.png)

### 统一内存预取
由于CUDA虚拟内存，可以放在任意设备内存上（CPU或者GPU），因此调用时会发生多次缺页，将内存中的数据搬到显存当中，造成内存瓶颈
迁移的开销很大，因为缺页是逐个发生的，可以通过预取，将内存中的数据预先搬运到显存当中
![alt text](image-4.png)

    cudaMemLocation location{};
    location.type = cudaMemLocationTypeDevice;
    location.id = 0;  // GPU 编号，单卡填 0

    cudaMemPrefetchAsync(x,n*sizeof(float),location,0);

### 调度顺序

<img src="image-6.png" alt="调度顺序" width="360">

粒度：一个warp32个线程是同时执行的（同一个 warp 的 lane 一起执行）

### 关于统一内存
区分：unified memory、global memory、shared memory

Shared Memory：
    片上高速存储，**block 内共享，必须显式使用。**

Global Memory：
    **显存**，全 GPU 可见，通过 L1/L2 缓存。

Unified Memory：
    统一虚拟地址和页面迁移机制，不是独立的物理存储层级。


统一内存是一个单一的内存地址空间，系统中的任**何处理器都可以访问**。cudaMallocManaged() 是一个分配函数，它返回一个可从任何处理器访问的指针（下文中为 ptr ）。
Unified Memory 用一个统一虚拟地址连接 CPU 和 GPU，让同一指针在两边都能使用；物理页面仍然位于 CPU 内存或 GPU 显存，并按页迁移。

传统的cudaMalloc需要两份指针和显示的复制：
```
float *h_x;
float *d_x;

cudaMallocHost(&h_x, size);
cudaMalloc(&d_x, size);

cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(d_x);
cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
```

Unified Memory使用一个指针
```
float *x;
cudaMallocManaged(&x, size);
```

当 GPU 访问一个当前位于 CPU 内存的页面时，可能发生：
1. GPU 触发缺页。
2. CUDA 驱动暂停相关访问。
3. 把页面从 CPU 内存迁移到 GPU 显存。
4. 更新页表。
5. GPU 继续访问。
CPU访问位于显存的页面也同样

cudaMallocManaged 只负责建立统一地址。
CPU 初始化后，页面通常先在主机内存。
cudaMemPrefetchAsync 把页面提前迁移到 GPU。
kernel 访问 GPU 本地页面时，性能才接近普通 global memory。
CPU 在 kernel 结束前访问同一块数据，会产生数据竞争或额外迁移。
cudaFree 负责释放 managed allocation。

## 0.2 关于global mem合并访存
| | Global Memory | Shared Memory |
|---|---|---|
| 位置 | 显卡显存，片外 | SM 片内 |
| 容量 | GB 级 | 每 block 通常几十 KB |
| 延迟 | 高，几百个周期 | 低，几十个周期 |
| 作用域 | 所有 block、所有线程 | **同一个 block 内**的线程 |
| 生命周期 | 从 `cudaMalloc` 到 `cudaFree` | block 执行期间 |
| 声明方式 | `cudaMalloc` 得到指针 | `__shared__` |
| 主要性能问题 | 是否合并访存 | 是否发生 bank conflict |
| 同步 | 跨 block 不能依赖顺序 | 常用 `__syncthreads()` |

shared mem只在SM内部共享，只在当前 block 内共享，不同 block 之间不能直接共享 shared memory；跨 block 通信还是要用 global memory、原子操作等

### global mem合并访存
global mem即是设备显存，可以被host和所有GPU线程访问
分配方式有
cudaMalloc();
cudaMallocManaged();
__device\__ float data[]

关于warp，假设一个线程块中有256个线程，每个warp有32个线程，即8个warp，SM 不会每次只取一个线程来执行，而是取一个 warp，为这个 warp 发射一条指令（SIMT，单指令多线程）。一个 SM 里可以同时驻留多个 warp，warp scheduler 会在它们之间切换。

在 CUDA 中访问全局内存时，需要考虑的一个关键方面是**同一 warp 内32个线程所访问的内存位置之间有何关系**。这些内存访问模式会直接影响内存访问效率和应用程序的整体性能。

全局内存通过 **32 字节**（称为sector）内存事务进行访问。当 CUDA 线程从全局内存请求数据时，**该 warp 中所有线程的内存访问会被合并为最少的内存事务数**。所需内存事务数取决于每个线程访问的字大小以及内存地址在各线程间的分布。
也就是说，是把各个warp请求的内存合并起来调度

意思就是，global mem的访问粒度是32B,一次最少搬运32B，而warp内的各个线程的读不是孤立的，如果落在同一个32B粒度内那就可以合并。如果地址相差太大，就会造成很明显的读放大

所谓合并访存，合并访存发生在**同一个 warp 的同一条 load/store 指令内**。硬件会看这个 warp 中所有 active lane 的地址，把它们**映射到尽量少的 sector**。不同 warp 之间不会合并。warp 内一条访存指令的地址会一起参与合并；地址集中在少量 32B sector 内时，传输量小；地址分散到很多 sector 时，虽然每个线程只需要 4B，实际却取回大量未使用的字节，形成读放大。
合并访存：
![alt text](image-7.png)
非合并访存：
![alt text](image-8.png)

### 实验
两个kernel,一个合并访存，一个不合并访存
调用ncu --section MemoryWorkloadAnalysis_Tables --print-details=all ./coalesced
可以看到，ncu已经给我们提醒了未合并访存的读放大和写放大问题
![alt text](image-9.png)

![alt text](image-10.png)

进一步分析DRAM数据：
ncu --metrics group:memory__dram_table ./coalesced
![alt text](image-12.png)
可以看到读放大为八倍

ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./coalesced
![alt text](image-13.png)
可以看到，合并访存的请求数：sector数量为1：4，因为请求按warp粒度进行，一个warp32个线程共请求32*4=128B，需要4个sector，符合预期
而不合并访存的请求数：sector数量为1：32，因为每个线程都需要读一个sector

回想之前的两种add实现：
```

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


```
![alt text](image-14.png)
也正好对应了1：4与1：32

## 0.3 Shared memory access in CUDA
不同层级的存储概览
| 存储 | 位置与作用域 | 本机容量 | 典型延迟 | 带宽定位 | 管理方式 |
|---|---|---|---:|---|---|
| 寄存器 | SM 内，线程私有，kernel 生命周期 | 每 SM 256 KiB | 约 1 个周期 | 最高 | 编译器分配 |
| Shared Memory | SM 内，同一 block 共享 | 100 KiB/SM | 约 20–40 周期 | 每 SM 约 128 B/周期，冲突时下降 | 程序员显式管理 |
| L1 Cache | SM 内，自动缓存 | 与 shared 共用物理存储 | 约 30–50 周期 | 与 shared 同一数量级 | 硬件自动 |
| L2 Cache | GPU 内，全卡共享 | 32 MiB | 约 100–300 周期 | 通常为 DRAM 的数倍以内 | 硬件自动，可设置 persistence |
| Global / DRAM | 片外显存 | 7.52 GiB 可用 | 约 300–600 ns | 448 GB/s 理论值 | `cudaMalloc`、访存合并 |
| Constant | 显存承载，常数缓存加速 | 64 KiB | 命中低，未命中到显存 | warp 同地址时广播 | `__constant__` |
| Local | 物理上在显存，线程私有 | 占用显存 | 命中 L1 较快，否则接近 DRAM | 低 | 编译器产生 spill 或大数组 |
| Host Memory | CPU 内存，经 PCIe 访问 | 系统内存 | 微秒级 | 约 32 GB/s 单向理论上限 | `cudaMemcpy`、prefetch |


### Shared memory
沿第二维及更高维度访问多维数组中的元素时，就无法合并global memory访存。不过，如果使用共享内存，在这种情况下就可以合并内存访问。

- shared memory的延迟量级与L1Cache同量级，是global的约100倍快（前提是不存在bank冲突）。
- shared memory**按照线程块分配，同一线程块中的所有线程可以访问同一块shared memory**，可以共享同一线程块中的线程的数据

##### 线程同步
虽然同一线程块中的不同线程共享shared memory，但是同一线程块中会包含多个warp，所以并不是同时执行的。假设出现写后读，就要进行线程同步。

同步原语：__syncthreads()
所有线程必须都执行了__syncthreads()才能越过__syncthreads()继续执行
**因此，可以在写后读的写与读之间，添加__syncthreads()**
前提是__syncthreads()前的代码不能是死循环，否则会死锁

#### 示例
只能在kernel内声明共享内存
使用 __shared__ 变量声明说明符声明共享内存
1. 动态分配：
    在kernel内部，使用```extern __shared__ int s[];```来进行动态分配
    在kernel启动时，启动方式为kernel<<<grid dim,block dim,**space**>>>(para)来启动
2. 静态分配：
    在kernel内部,使用```__shared__  int[64]```来静态分配
    启动方式为直接```kernel<<<grid,block>>>(para)```

#### bank conflict
bank conflict只会发生在：
- 同一个warp
- 同一个条shared mem load/store指令
- 同一个warp中不同thread对不同shared memory不同地址的访问

shared memory最大的带宽为：32 个 bank × 4 Byte/bank/周期 = 128 Byte/周期/SM
bank地址为：bank = word ％ 32

广播和多播：如果不同的thread访问的是同一个bank中的同一个地址，可以进行广播和多播，主要面向读访问

| shared memory 访问 | bank 使用情况 | 冲突度 | 有效带宽 |
|---|---|---:|---:|
| `temp[tid]` | 32 个线程分别使用 32 个 bank | 1 | 约 128 B/周期 |
| `temp[2 * tid]` | 只有偶数 bank，每个使用两次 | 2 | 约 64 B/周期 |
| `temp[32 * tid]` | 全部使用 bank 0 | 32 | 约 4 B/周期 |
| 所有线程读 `temp[0]` | 同一个地址 | 广播 | 一次读取后广播 |
| `temp[tid]` 的 64 线程版本 | 每个 warp 各自覆盖 32 个 bank | 1 | 两个 warp 分别执行 |

#### 实验：
![alt text](image-17.png)
可以看到，即使bank conflict，kernel的时间也没有拉开差距
因为conflict对比非conflict：
非conflict有2个warp，即两次写+两次读
conflict有2个warp,即2 * 32次写+2 * 32次读
两者相差128-4=124个周期，频率1.5～3GHz，则只相差约40～80ns

![alt text](image-18.png)
可以看到，放大conflict量级，差距很大
粗略估算：每个warp增加31个额外wavefront（波前，一个 warp 的访存请求经过 LSU/L1TEX 流水线时，被硬件分成的一批处理单元。），共blocks * warps = 216 * 8 = 1728 warps
会造成1728 * 31 * N=53568N个额外的wavefront
36SM下额外时间为53568N/36=1488N个周期

N=100000，频率最大为3.09GHz，则增加的时间最少为1488*100000/3.09=48,155,339ns
实际增加为50,880,854 ns，结果非常吻合，当前频率约为2.924 GHz
