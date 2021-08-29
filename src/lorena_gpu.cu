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

#define EPSILON 0.1
#define PI 3.14159265358979323846

__global__ void modifica_A_B(int *adjlist,int k,double a, double b,double *A,double *B,int size){
    int n=threadIdx.x+blockDim.x*blockIdx.x;

    if(n<size){
        A[n]+=a*adjlist[k*size+n];
        B[n]+=b*adjlist[k*size+n];
    }
}

/*
Create a partition and sum to get the cost of the cut
*/
__global__ void partitionAndSum(int *adjmat,int *partitions,double alfa,double *teta,int size){
    int n=threadIdx.x+blockDim.x*blockIdx.x;

    int smem[THREADS_PER_BLOCK];
    if(n<size){
        /*Next instructions equivalent to the following conditional statement
        smem[i] = ((teta[n] >= alfa) && (teta[n] < alfa+PI)) ? 1 : -1;
        but do not introduce divergence
        */
        smem[threadIdx.x]=(int)((teta[n] >= alfa) && (teta[n] < alfa+PI))*2-1;
    
    }

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
    gpuErrCheck(cudaMallocHost((void**)&A_cpu,g->getSize()*sizeof(double)));
    gpuErrCheck(cudaMalloc((void**)&A_gpu,g->getSize()*sizeof(double)));
    /*gpuErrCheck(cudaHostAlloc((void**)&A_cpu,g->getSize()*sizeof(double),cudaHostAllocMapped));
    gpuErrCheck(cudaHostGetDevicePointer((void**)&A_gpu,(void*)A_cpu,0));*/
    gpuErrCheck(cudaMallocHost((void**)&B_cpu,g->getSize()*sizeof(double)));
    gpuErrCheck(cudaMalloc((void**)&B_gpu,g->getSize()*sizeof(double)));
    /*gpuErrCheck(cudaHostAlloc((void**)&B_cpu,g->getSize()*sizeof(double),cudaHostAllocMapped));
    gpuErrCheck(cudaHostGetDevicePointer((void**)&B_gpu,(void*)B_cpu,0));*/
    
	for (int i=0; i<size; i++) {
		A_cpu[i] = B_cpu[i] = 0;
		for(int j=0; j<size; j++) {
			A_cpu[i] += g->getAdjmat()[i][j]*cos(teta[j]);
			B_cpu[i] += g->getAdjmat()[i][j]*sin(teta[j]);
		}
	}
    
    gpuErrCheck(cudaMemcpy(A_gpu,A_cpu,size*sizeof(double),cudaMemcpyHostToDevice));
    gpuErrCheck(cudaMemcpy(B_gpu,B_cpu,size*sizeof(double),cudaMemcpyHostToDevice));


    bool end=false;
    
    int n_blocks=(int)(g->getSize()+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    
	double p,alfa;
    
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

            //gpuErrCheck(cudaDeviceSynchronize());
            gpuErrCheck(cudaMemcpy(A_cpu,A_gpu,size*sizeof(double),cudaMemcpyDeviceToHost));
            gpuErrCheck(cudaMemcpy(B_cpu,B_gpu,size*sizeof(double),cudaMemcpyDeviceToHost));

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
