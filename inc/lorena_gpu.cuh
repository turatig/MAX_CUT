#ifndef LORENA_GPU
#define LORENA_GPU
#include "Graph.cuh"
__global__ void modifica_A_B(int *adjlist,int k,double a, double b,double *A,double *B,int size);
__global__ void makePartition(int *partition,double alfa,double *teta,int size);
__global__ void cutCost(int *adjmat,int *partition,double *res,int size);
int *maximumCut(Graph *g,double *teta);
double *circleMap(Graph *g,double *_teta);
#endif
