/*
Cuda kernel and functions used to optimize a Hopefield network while solving the max_cut problem
*/

/*
Update the status of the i-th node in the graph by summing
*/
#include <stdio.h>
#include <iostream>
#include "../inc/Graph.cuh"
#include "../inc/utils.cuh"

#define THREADS_PER_BLOCK 256

__global__ void statusUpdate(int *adjmat,int node,int *status,int size,int *res){
    /*declare smem to have a size equal to the double of the max number of threads per block*/
    __shared__ int smem[THREADS_PER_BLOCK];
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    int i;

    if(n<size)  smem[threadIdx.x]=-(adjmat[node*size+n]*status[n]);
    else smem[threadIdx.x]=0;

    __syncthreads();
    
    for(i=2;blockDim.x/i>32;i<<=1){
        if(threadIdx.x<blockDim.x/i) smem[threadIdx.x]+=smem[threadIdx.x+blockDim.x/i];
        __syncthreads();
    }

    /*
    Unrolling of the last five iterations (when only one warp is invloved) of the cycle to improve efficiency
    */
    if(threadIdx.x<blockDim.x/i) smem[threadIdx.x]+=smem[threadIdx.x+blockDim.x/i];
     __syncthreads();
     i<<=1;
    if(threadIdx.x<blockDim.x/i) smem[threadIdx.x]+=smem[threadIdx.x+blockDim.x/i];
     __syncthreads();
     i<<=1;
    if(threadIdx.x<blockDim.x/i) smem[threadIdx.x]+=smem[threadIdx.x+blockDim.x/i];
     __syncthreads();
     i<<=1;
    if(threadIdx.x<blockDim.x/i) smem[threadIdx.x]+=smem[threadIdx.x+blockDim.x/i];
     __syncthreads();
     i<<=1;
    if(threadIdx.x<blockDim.x/i) smem[threadIdx.x]+=smem[threadIdx.x+blockDim.x/i];
     __syncthreads();
     i<<=1;
    if(threadIdx.x<blockDim.x/i) smem[threadIdx.x]+=smem[threadIdx.x+blockDim.x/i];
     __syncthreads();

    if(threadIdx.x==0) res[blockIdx.x]=smem[0];
}

int *stabilizeHopfieldNet(Graph g){
    int *status;
    int *status_cpu;
    int *res;
    int *res_cpu;
    int n,prev,count=0;
    int *adjmat=g.getDevicePointer();
    //double start,elapsed;
    

    gpuErrCheck(cudaMalloc((void**)&status,g.getSize()*sizeof(int)));
    gpuErrCheck(cudaMallocHost((void**)&status_cpu,g.getSize()*sizeof(int)));

    for(int i=0;i<g.getSize();i++) status_cpu[i]=0;
    gpuErrCheck(cudaMemcpy(status,status_cpu,g.getSize()*sizeof(int),cudaMemcpyHostToDevice));

    bool end=false;
    
    int n_blocks=(int)(g.getSize()+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    gpuErrCheck(cudaMalloc((void**)&res,n_blocks*sizeof(int)));
    gpuErrCheck(cudaMallocHost((void**)&res_cpu,n_blocks*sizeof(int)));
    
    while(!end){
        end=true;
        for(int i=0;i<g.getSize();i++){
            
           /* std::cout<<"Adjacency list of the node "<<i<<"\n";
            for(int k=0;k<g.getSize();k++) std::cout<<g.getAdjmat()[i][k]<<" ";
            std::cout<<"\n";
            std::cout<<"Status of the network\n";
            for(int k=0;k<g.getSize();k++)std::cout<<status[k]<<" ";
            std::cout<<"\n";
            for(int k=0;k<n_blocks;k++) res[k]=0;
            for(int k=0;k<n_blocks;k++) std::cout<<res[k]<<" ";
            std::cout<<"\n";*/

            /*Compute sum reduction on device*/
            //std::cout<<"Calling kernel\n";
            //start=cpuSecond();
            statusUpdate<<<n_blocks,THREADS_PER_BLOCK>>>(adjmat,i,status,g.getSize(),res);
            gpuErrCheck(cudaDeviceSynchronize());
            //elapsed=cpuSecond()-start;
            //std::cout<<std::fixed<<"GPU reduction took "<<elapsed<<"sec\n";
            //std::cout<<"Kernel terminated successfully\n";
            
            /*Sum partial results together*/
            n=0;
            /*std::cout<<"n_blocks "<<n_blocks<<"\n";
            std::cout<<"Sum "<<res[0]<<"\n";*/
            gpuErrCheck(cudaMemcpy(res_cpu,res,n_blocks*sizeof(int),cudaMemcpyDeviceToHost));
            for(int j=0;j<n_blocks;j++) n+=res_cpu[j];
            //std::cout<<"Sum="<<n<<"\n";
            prev=status_cpu[i];

            if(n<0) status_cpu[i]=-1;
            else status_cpu[i]=1;

            if(status_cpu[i]!=prev) end=false;

            gpuErrCheck(cudaMemcpy(&status[i],&status_cpu[i],sizeof(int),cudaMemcpyHostToDevice));
        }
        /*std::cout<<"-------ITERATION "<<count<<"-------\n";
        std::cout<<"End of the cycle: "<<end<<"\n";*/
    }
    cudaFree(res);
    cudaFree(status);
    
    cudaFreeHost(res_cpu);

    return status_cpu;
}
