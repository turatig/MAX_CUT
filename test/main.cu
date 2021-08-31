#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../inc/rete_cpu.cuh"
#include "../inc/rete_gpu.cuh"
#include "../inc/lorena_cpu.cuh"
#include "../inc/lorena_gpu.cuh"
#include "../inc/Graph.cuh"
#include "../inc/utils.cuh"


#define PI 3.14159265358979323846

/*
Driver program to test parallel implementation of the algorithms against sequential one.
Call main size seed sparsity.
    -size: the number of nodes in the graph
    -seed: to reproduce the experiment
    -sparsity: percentage of nodes connected (given as integer parameter e.g. 50 means approximately 50% of nodes connected)
*/
int main(int argc,char **argv){
    double start,elapsed;
    int size, sparsity,seed;
    if(argc<2){
        std::cout<<"The size of the graph must be specified\n";
        return -1;
    }
    size=atoi(argv[1]);
    if(argc>=3){
        seed=atoi(argv[2]);
        srand(seed);
    }
    else
       srand(time(NULL));
    if(argc>=4)
        sparsity=atoi(argv[3]);
    else
        sparsity=rand()%100;

    std::cout<<"Size: "<<size<<" Seed: "<<seed<<" Sparsity: "<<sparsity<<"\n";
    Graph *g=new Graph(size,sparsity);
    start=cpuSecond();
    int * status_cpu=rete_cpu::stabilizza_rete_Hopfield(g->getAdjmat(),g->getSize());
    elapsed=cpuSecond()-start;
    std::cout<<"Hopfield: sequential implementation ended in "<<elapsed<<" sec\n";
    
    start=cpuSecond();
    int *status_gpu=stabilizeHopfieldNet(g);
    elapsed=cpuSecond()-start;
    std::cout<<"Hopfield: parallel implementation ended in "<<elapsed<<" sec\n";

    std::cout<<"\n";
    if(check_output(status_cpu,status_gpu,g->getSize()))
        std::cout<<"------SUCCESS------\n";
    else
        std::cout<<"------ERROR: PARALLEL OUTPUT MUST AGREE WITH SEQUENTIAL ONE------\n";

    free(status_cpu);
    cudaFreeHost(status_gpu);
    
    double *teta=(double*)malloc(g->getSize()*sizeof(double));
	for (int i=0; i<g->getSize(); i++)
		teta[i] = (double) 2*PI*rand()/RAND_MAX;
    
    //Execution time is logged in the function
    status_cpu=lorena_cpu::mapAndCut(g->getAdjmat(),teta,g->getSize());

    start=cpuSecond();
    double *updated_teta=circleMap(g,teta);
    elapsed=cpuSecond()-start;
    std::cout<<"Lorena--mapping points: parallel implementation ended in "<<elapsed<<" sec\n";

    start=cpuSecond();
    status_gpu=maximumCut(g,updated_teta);
    elapsed=cpuSecond()-start;
    std::cout<<"Lorena--find best partition: parallel implementation ended in "<<elapsed<<" sec\n";


    if(check_output(status_cpu,status_gpu,g->getSize()))
        std::cout<<"------SUCCESS------\n";
    else
        std::cout<<"------ERROR: PARALLEL OUTPUT MUST AGREE WITH SEQUENTIAL ONE------\n";
    

    free(status_cpu);
    free(status_gpu);
    free(teta);
    free(updated_teta);
    delete g;

}
