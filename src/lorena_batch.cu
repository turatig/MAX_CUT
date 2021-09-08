#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "../inc/Graph.cuh"
#include "../inc/lorena_batch.cuh"
#include "../inc/utils.cuh"

#define THREADS_PER_BLOCK 256
#define PI 3.14159265358979323846
/*
-- Alternative version of the cut procedure --
   Main idea: create and evaluate partition in parallel
*/
/*
Same as lorena_gpu/makePartition -- the only difference is that this is done in parallel for the whole batch
        - Call for (size*batch_size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK
*/
__global__ void makePartitionsBatch(int *partitions,int oset,int batch_size,double *teta,int size){
    int n=threadIdx.x+blockDim.x*blockIdx.x;
    int k=oset+(int)n/size;
    double alfa;

    if(n<size*batch_size && k<size){
		alfa = (int)(teta[k] > PI)*(teta[k]-PI) + (int)(teta[k] <= PI)*teta[k];
        /*Next instructions equivalent to the following conditional statement
        smem[i] = ((teta[n] >= alfa) && (teta[n] < alfa+PI)) ? 1 : -1;
        but do not introduce divergence
        */
        partitions[n]=(int)((teta[n%size] >= alfa) && (teta[n%size] < alfa+PI))*2-1;
    
    }

}
/*
Same as lorena_gpu/cutCost -- the only difference is that partial results array is now interleaved for the whole batch
*/
__global__ void cutCostBatch(int *adjmat,int *partitions,int *res,int res_size,int batch_size,int size){
    int n=threadIdx.x+blockDim.x*blockIdx.x;
    __shared__ int smem[THREADS_PER_BLOCK];
    int z=(int)n/size;
    int j=n%size;
    int i;
    
    for(int k=0;k<batch_size;k++){
        if(n<size*size)
            //Putting data in the shared memory
            smem[threadIdx.x]=adjmat[z*size+j]*(1-partitions[k*size+z]*partitions[k*size+j]);
        else
            smem[threadIdx.x]=0;

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

        if(threadIdx.x==0) res[blockIdx.x*batch_size+k]=smem[0];
        __syncthreads();
    }
}

/*
Sum partial results computed with previously defined kernels and max reduce to find the best partition in batch
    - partial results array is interleaved for the batch to guarantee coalescent access
    - the maximum partition for the batch will be at partitions[best_idx]
*/

__global__ void maxInBatch(int *res,int res_size,int batch_size,int *best_idx,long *best_cost){
    int n=threadIdx.x;
    
    int i;
    //This variables are used to compute max reduction condition and store the evaluation before smem[n] is changed
    int sx,dx;
    __shared__ long smem[THREADS_PER_BLOCK];
    __shared__ int smem_idx[THREADS_PER_BLOCK];
    long s=0;

    if(n<batch_size){
        for(int k=0;k<res_size;k++){s+=res[k*batch_size+n];}

    }

    smem[n]=s/4;
    smem_idx[n]=n;
    __syncthreads();

    //Max reduction of the smem array to find the partition that maximize cost function in the batch
    for(i=blockDim.x>>1;i>32;i>>=1){
        /*
         This is equivalent to smem[n]= smem[n]>=smem[n+i] ? smem[n] : smem[n+i] but this does not introduce branch divergence
        */
        if(n<i){
            sx=(int)(smem[n]>smem[n+i]);
            dx=(int)(smem[n]<=smem[n+i]);
            smem[n]=sx*smem[n]+dx*smem[n+i];
            smem_idx[n]=sx*smem_idx[n]+dx*smem_idx[n+i];
        }
        __syncthreads();
    }

    /*
    Unrolling of the last five iterations (when only one warp is invloved) of the cycle to improve efficiency
    */
    if(n<i){
        sx=(int)(smem[n]>smem[n+i]);
        dx=(int)(smem[n]<=smem[n+i]);
        smem[n]=sx*smem[n]+dx*smem[n+i];
        smem_idx[n]=sx*smem_idx[n]+dx*smem_idx[n+i];
    }
    __syncthreads();
    i>>=1;

    if(n<i){
        sx=(int)(smem[n]>smem[n+i]);
        dx=(int)(smem[n]<=smem[n+i]);
        smem[n]=sx*smem[n]+dx*smem[n+i];
        smem_idx[n]=sx*smem_idx[n]+dx*smem_idx[n+i];
    }
    __syncthreads();
    i>>=1;

    if(n<i){
        sx=(int)(smem[n]>smem[n+i]);
        dx=(int)(smem[n]<=smem[n+i]);
        smem[n]=sx*smem[n]+dx*smem[n+i];
        smem_idx[n]=sx*smem_idx[n]+dx*smem_idx[n+i];
    }
    __syncthreads();
    i>>=1;

    if(n<i){
        sx=(int)(smem[n]>smem[n+i]);
        dx=(int)(smem[n]<=smem[n+i]);
        smem[n]=sx*smem[n]+dx*smem[n+i];
        smem_idx[n]=sx*smem_idx[n]+dx*smem_idx[n+i];
    }
    __syncthreads();
    i>>=1;

    if(n<i){
        sx=(int)(smem[n]>smem[n+i]);
        dx=(int)(smem[n]<=smem[n+i]);
        smem[n]=sx*smem[n]+dx*smem[n+i];
        smem_idx[n]=sx*smem_idx[n]+dx*smem_idx[n+i];
    }
    __syncthreads();
    i>>=1;

    if(n<i){
        sx=(int)(smem[n]>smem[n+i]);
        dx=(int)(smem[n]<=smem[n+i]);
        smem[n]=sx*smem[n]+dx*smem[n+i];
        smem_idx[n]=sx*smem_idx[n]+dx*smem_idx[n+i];
    }
    __syncthreads();
    
    if(n==0){
        *best_idx=smem_idx[n];
        *best_cost=smem[n];
    }

}

int *maximumCutBatch(Graph *g,double *teta,int batch_size){
    int *adjmat=g->getDevicePointer();
    int size=g->getSize();
    long max_cost=0,cost,*best_cost;
    int max_idx,*best_idx;
    int *partition,*partitions,*partitions_cpu,*res;
    double *teta_gpu;
    int res_size=(int)((size*size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK);
    int n_blocks; 

    gpuErrCheck(cudaMalloc((void**)&partitions,batch_size*size*sizeof(int*)));
    gpuErrCheck(cudaMemset(partitions,0,size*batch_size*sizeof(int)));
    partitions_cpu=(int*)malloc(size*batch_size*sizeof(int));
    partition=(int*)malloc(size*sizeof(int));

    gpuErrCheck(cudaMalloc((void**)&teta_gpu,size*sizeof(double)));
    gpuErrCheck(cudaMemcpy(teta_gpu,teta,size*sizeof(double),cudaMemcpyHostToDevice));

    gpuErrCheck(cudaMalloc((void**)&best_cost,sizeof(long)));
    gpuErrCheck(cudaMalloc((void**)&best_idx,sizeof(int)));
    
    gpuErrCheck(cudaMalloc((void**)&res,res_size*batch_size*sizeof(int)));

    for(int i=0;i<size;i+=batch_size){
        //__global__ void makePartitionsBatch(int *partitions,int oset,int batch_size,double *teta,int size){

        n_blocks=(size*batch_size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        makePartitionsBatch<<<n_blocks,THREADS_PER_BLOCK>>>(partitions,i,batch_size,teta_gpu,size);
        gpuErrCheck(cudaDeviceSynchronize());

        //__global__ void cutCostBatch(int *adjmat,int *partitions,int *res,int res_size,int batch_size,int size){
        n_blocks=(size*size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        cutCostBatch<<<n_blocks,THREADS_PER_BLOCK>>>(adjmat,partitions,res,res_size,batch_size,size);
        gpuErrCheck(cudaDeviceSynchronize());
        
        //__global__ void maxInBatch(int *res,int res_size,int batch_size,int best_idx,long best_cost){
        n_blocks=(batch_size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        maxInBatch<<<n_blocks,THREADS_PER_BLOCK>>>(res,res_size,batch_size,best_idx,best_cost);
        gpuErrCheck(cudaDeviceSynchronize());

        gpuErrCheck(cudaMemcpy(&max_idx,best_idx,sizeof(int),cudaMemcpyDeviceToHost));
        gpuErrCheck(cudaMemcpy(&cost,best_cost,sizeof(long),cudaMemcpyDeviceToHost));

        if(cost>max_cost){
            max_cost=cost;
            gpuErrCheck(cudaMemcpy(partitions_cpu,partitions,size*batch_size*sizeof(int),cudaMemcpyDeviceToHost));
            memcpy(partition,partitions_cpu+max_idx*size,size*sizeof(int));
        }
        
    }
    std::cout<<"Lorena (parallel)->"<<max_cost<<"\n";
    cudaFree(res);
    cudaFree(partitions);
    cudaFree(teta_gpu);
    cudaFree(best_cost);
    cudaFree(best_idx);
    free(partitions_cpu);

    return partition;
}
