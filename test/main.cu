#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../inc/graph_gen.h"
#include "../inc/rete_cpu.cuh"
#include "../inc/rete_gpu.cuh"
#include "../inc/Graph.cuh"
#include "../inc/utils.cuh"

/*
Function used to check the correctness of the parallel algorithm with respect to its sequential implementation
*/
bool check_output(int *seq_partitions,int *par_partitions,int size){
    for(int i=0;i<size;i++){
        if(par_partitions[i]!=seq_partitions[i]) return false;
    }
    return true;
}

int main(){
    /*Logging gpu infos*/
    /*int dev;
    cudaDeviceProp props;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props,dev);
    std::cout<<"Compute capability "<<props.major<<"."<<props.minor<<"\n";*/
    double start,elapsed;

    Graph g=Graph(10000,50);
    inizializza_strutture(g.getAdjmat(),g.getSize());
    start=cpuSecond();
    stabilizza_rete_Hopfield();
    elapsed=cpuSecond()-start;
    std::cout<<"Sequential implementation ended in "<<elapsed<<" sec\n";
    
    start=cpuSecond();
    int *status=stabilizeHopfieldNet(g);
    elapsed=cpuSecond()-start;
    std::cout<<"Parallel implementation ended in "<<elapsed<<" sec\n";

    std::cout<<"\n";
    if(check_output(get_stato_rete(),status,g.getSize()))
        std::cout<<"------SUCCESS------\n";
    else
        std::cout<<"------ERROR: PARALLEL OUTPUT MUST AGREE WITH SEQUENTIAL ONE------\n";
}
