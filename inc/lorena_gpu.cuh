#ifndef LORENA_GPU
#define LORENA_GPU
#include "Graph.cuh"
__global__ void modifica_A_B(int *adjlist,int k,double a, double b,double *A,double *B,int size);
__global__ void partitionAndSum(int *adjmat,int *partitions,double alfa,double *teta,int size);
double *circleMap(Graph *g,double *_teta);
#endif
