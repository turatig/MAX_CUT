/*
Cuda kernel and functions used to compute the lorena algorithm
*/

/*
Update the status of cos/sin vectors during the mapping phase
*/
#include <stdio.h>
#include <iostream>
#include "../inc/lorena_gpu.cuh"
#include "../inc/Graph.cuh"
#include "../inc/utils.cuh"

#define THREADS_PER_BLOCK 256
#define Min(x,y) (fabs(x) < fabs(y) ? fabs(x) : fabs(y))

#define EPSILON 5
#define PI 3.14159265358979323846

/*
Update cos and sin for every node's angle depending on node's k adjiacency list
*/
__global__ void modifica_A_B(int *adjlist,int k,double a, double b,double *A,double *B,int size){
    int n=threadIdx.x+blockDim.x*blockIdx.x;

    if(n<size){
        A[n]+=a*adjlist[k*size+n];
        B[n]+=b*adjlist[k*size+n];
    }
}

/*
Create a partition given a value for alpha
*/
__global__ void makePartition(int *partition,double alfa,double *teta,int size){
    int n=threadIdx.x+blockDim.x*blockIdx.x;

    if(n<size){
        /*Next instructions equivalent to the following conditional statement
        smem[i] = ((teta[n] >= alfa) && (teta[n] < alfa+PI)) ? 1 : -1;
        but do not introduce divergence
        */
        partition[n]=(int)((teta[n] >= alfa) && (teta[n] < alfa+PI))*2-1;
    
    }

}
/*
Compute the cost of partition
*/
__global__ void cutCost(int *adjmat,int *partition,int *res,int size){
    int n=threadIdx.x+blockDim.x*blockIdx.x;
    __shared__ int smem[THREADS_PER_BLOCK];
    int i=(int)n/size;
    int j=n%size;

    if(n<size*size)
        //Putting data in the shared memory
        smem[threadIdx.x]=adjmat[i*size+j]*(1-partition[i]*partition[j]);
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

    if(threadIdx.x==0) res[blockIdx.x]=smem[0];
}


double *circleMap(Graph *g,double *_teta){
    double *A_cpu,*B_cpu,*teta;
    double *A_gpu,*B_gpu;
    int *adjmat=g->getDevicePointer();
    int size=g->getSize();

    teta=(double*)malloc(size*sizeof(double));
    /*Hard copy value of teta to avoid external changes*/
    memcpy(teta,_teta,size*sizeof(double));

    /*
    Alloc vectors of cos/sin(teta[i])[A,B] and results array as zero-copy memory to avoid explicit transfer while looping
    */
    /*gpuErrCheck(cudaMallocHost((void**)&A_cpu,g->getSize()*sizeof(double)));
    gpuErrCheck(cudaMalloc((void**)&A_gpu,g->getSize()*sizeof(double)));*/
    gpuErrCheck(cudaHostAlloc((void**)&A_cpu,g->getSize()*sizeof(double),cudaHostAllocMapped));
    gpuErrCheck(cudaHostGetDevicePointer((void**)&A_gpu,(void*)A_cpu,0));
    /*gpuErrCheck(cudaMallocHost((void**)&B_cpu,g->getSize()*sizeof(double)));
    gpuErrCheck(cudaMalloc((void**)&B_gpu,g->getSize()*sizeof(double)));*/
    gpuErrCheck(cudaHostAlloc((void**)&B_cpu,g->getSize()*sizeof(double),cudaHostAllocMapped));
    gpuErrCheck(cudaHostGetDevicePointer((void**)&B_gpu,(void*)B_cpu,0));
    
	for (int i=0; i<size; i++) {
		A_cpu[i] = B_cpu[i] = 0;
		for(int j=0; j<size; j++) {
			A_cpu[i] += g->getAdjmat()[i][j]*cos(teta[j]);
			B_cpu[i] += g->getAdjmat()[i][j]*sin(teta[j]);
		}
	}
    
    //gpuErrCheck(cudaMemcpy(A_gpu,A_cpu,size*sizeof(double),cudaMemcpyHostToDevice));
    //gpuErrCheck(cudaMemcpy(B_gpu,B_cpu,size*sizeof(double),cudaMemcpyHostToDevice));


    bool end=false;
    
    int n_blocks=(int)(g->getSize()+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    
	double alfa;
    
	while(!end ) {
		end = true;
		for (int k=0; k<size; k++) {
			alfa = teta[k];
			teta[k] = atan(B_cpu[k]/A_cpu[k]); 
			if (A_cpu[k] >= 0) 
				teta[k] += PI; 
			else if (B_cpu[k] > 0) 
				teta[k] += 2*PI;
			modifica_A_B<<<n_blocks,THREADS_PER_BLOCK>>>
                        (adjmat,k, cos(teta[k])-cos(alfa), sin(teta[k])-sin(alfa),A_gpu,B_gpu,size);

            gpuErrCheck(cudaDeviceSynchronize());
            /*gpuErrCheck(cudaMemcpy(A_cpu,A_gpu,size*sizeof(double),cudaMemcpyDeviceToHost));
            gpuErrCheck(cudaMemcpy(B_cpu,B_gpu,size*sizeof(double),cudaMemcpyDeviceToHost));*/

			if ( Min(alfa-teta[k],2*PI-alfa+teta[k]) > EPSILON )
				end = false;
		}
	}
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    
    cudaFreeHost(A_cpu);
    cudaFreeHost(B_cpu);

    return teta;
}

/*
Parallelization of the function "taglio_massimo" in lorena_cpu.cu
*/
int *maximumCut(Graph *g,double *teta){
    int *adjmat=g->getDevicePointer();
    int size=g->getSize();
    double alfa=0.0,max_alfa;
    int *partition,*res_cpu,*res_gpu;
    int n_blocks;
    long s=0,max_cost=0;
    double *teta_gpu;
    int res_size=(int)((size*size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK);

    gpuErrCheck(cudaMalloc((void**)&partition,size*sizeof(int)));
    gpuErrCheck(cudaMalloc((void**)&teta_gpu,size*sizeof(double)));
    gpuErrCheck(cudaMemcpy(teta_gpu,teta,size*sizeof(double),cudaMemcpyHostToDevice));
    gpuErrCheck(cudaMallocHost((void**)&res_cpu,res_size*sizeof(int)));
    gpuErrCheck(cudaMalloc((void**)&res_gpu,res_size*sizeof(int)));

    for(int i=0;i<size;i++){
		alfa = (teta[i] > PI) ? teta[i]-PI : teta[i];
        n_blocks=(int)(size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

        makePartition<<<n_blocks,THREADS_PER_BLOCK>>>(partition,alfa,teta_gpu,size);
        gpuErrCheck(cudaDeviceSynchronize());

        n_blocks=res_size;
        cutCost<<<n_blocks,THREADS_PER_BLOCK>>>(adjmat,partition,res_gpu,size);
        gpuErrCheck(cudaMemcpy(res_cpu,res_gpu,res_size*sizeof(int),cudaMemcpyDeviceToHost));
        
        s=0;
        for(int j=0;j<res_size;j++)
            s+=res_cpu[j];
        s/=4;
        if(s>max_cost){
            max_cost=s;
            max_alfa=alfa;
        }
    }
    std::cout<<"Lorena (parallel)->"<<max_cost<<"\n";
    int *res=(int*)malloc(size*sizeof(int));
    makePartition<<<n_blocks,THREADS_PER_BLOCK>>>(partition,max_alfa,teta_gpu,size);
    gpuErrCheck(cudaMemcpy(res,partition,size*sizeof(int),cudaMemcpyDeviceToHost));

    cudaFree(res_gpu);
    cudaFree(partition);
    cudaFree(teta_gpu);
    cudaFree(res_cpu);

    return res;
}

