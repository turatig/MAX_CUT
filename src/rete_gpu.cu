/*
Cuda kernel and functions used to optimize a Hopefield network while solving the max_cut problem
*/

#include <stdio.h>
#include <iostream>
#include "../inc/rete_gpu.cuh"
#include "../inc/Graph.cuh"
#include "../inc/utils.cuh"

#define THREADS_PER_BLOCK 256

/*
Update the status of i-th by summing status of node n weighted by weight in adjmat[i][n].
    -adjmat: adjacency matrix
    -node: idx of the i-th node
    -status: status vector
    -size: number of nodes in the graph
    -res: partial results array. Sum is reduced in shared memory, so that partial results must summed outside the function.
*/
__global__ void statusUpdate(int *adjmat,int node,int *status,int size,int *res){
    /*declare smem to have a size equal to the double of the max number of threads per block*/
    __shared__ int smem[THREADS_PER_BLOCK];
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    int i;

    if(n<size)  smem[threadIdx.x]=-(adjmat[node*(size)+n]*status[n]);
    else smem[threadIdx.x]=0;

    __syncthreads();
    
    for(i=blockDim.x>>1;i>32;i>>=1){
        if(threadIdx.x<i) smem[threadIdx.x]+=smem[threadIdx.x+i];
        __syncthreads();
    }

    /*
    Unrolling of the last five iterations (when only one warp is invloved) of the cycle to improve efficiency
    */
    if(threadIdx.x<i) smem[threadIdx.x]+=smem[threadIdx.x+i];
     __syncthreads();
     i>>=1;
    if(threadIdx.x<i) smem[threadIdx.x]+=smem[threadIdx.x+i];
     __syncthreads();
     i>>=1;
    if(threadIdx.x<i) smem[threadIdx.x]+=smem[threadIdx.x+i];
     __syncthreads();
     i>>=1;
    if(threadIdx.x<i) smem[threadIdx.x]+=smem[threadIdx.x+i];
     __syncthreads();
     i>>=1;
    if(threadIdx.x<i) smem[threadIdx.x]+=smem[threadIdx.x+i];
     __syncthreads();
     i>>=1;
    if(threadIdx.x<i) smem[threadIdx.x]+=smem[threadIdx.x+i];
     __syncthreads();

    if(threadIdx.x==0) res[blockIdx.x]=smem[0];
}

/*
Call gpu neurons update until the network is stable (i.e. no more neurons have changed their status)
    -g: graph object
*/
int *stabilizeHopfieldNet(Graph *g){
    int *status;
    int *status_cpu;
    int *res;
    int *res_cpu;
    int n,prev;
    int *adjmat=g->getDevicePointer();
    
    
    /*
    Alloc status and results array as zero-copy memory to avoid explicit transfer while looping
    */
    /*gpuErrCheck(cudaMallocHost((void**)&status_cpu,g->getSize()*sizeof(int)));
    gpuErrCheck(cudaMalloc((void**)&status,g->getSize()*sizeof(int)));*/
    gpuErrCheck(cudaHostAlloc((void**)&status_cpu,g->getSize()*sizeof(int),cudaHostAllocMapped));
    gpuErrCheck(cudaHostGetDevicePointer((void**)&status,(void*)status_cpu,0));

    for(int i=0;i<g->getSize();i++) status_cpu[i]=0;
    //gpuErrCheck(cudaMemcpy(status,status_cpu,g->getSize()*sizeof(int),cudaMemcpyHostToDevice));

    bool end=false;
    
    int n_blocks=(int)(g->getSize()+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    /*gpuErrCheck(cudaMalloc((void**)&res,n_blocks*sizeof(int)));
    gpuErrCheck(cudaMallocHost((void**)&res_cpu,n_blocks*sizeof(int)));*/
    gpuErrCheck(cudaHostAlloc((void**)&res_cpu,n_blocks*sizeof(int),cudaHostAllocMapped));
    gpuErrCheck(cudaHostGetDevicePointer((void**)&res,(void*)res_cpu,0));
    
    while(!end){
        end=true;
        for(int i=0;i<g->getSize();i++){
            
            statusUpdate<<<n_blocks,THREADS_PER_BLOCK>>>(adjmat,i,status,g->getSize(),res);
            gpuErrCheck(cudaDeviceSynchronize());
            
            /*Sum partial results together*/
            n=0;
            //gpuErrCheck(cudaMemcpy(res_cpu,res,n_blocks*sizeof(int),cudaMemcpyDeviceToHost));
            for(int j=0;j<n_blocks;j++) n+=res_cpu[j];
            prev=status_cpu[i];

            if(n<0) status_cpu[i]=-1;
            else status_cpu[i]=1;

            if(status_cpu[i]!=prev) end=false;

           //gpuErrCheck(cudaMemcpy(&status[i],&status_cpu[i],sizeof(int),cudaMemcpyHostToDevice));
        }
    }
    cudaFree(res);
    cudaFree(status);
    
    cudaFreeHost(res_cpu);

    return status_cpu;
}
