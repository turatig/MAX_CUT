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

int main(){
    double start,elapsed;

    Graph *g=new Graph(5000,50);
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
    
    start=cpuSecond();
    status_cpu=lorena_cpu::mapAndCut(g->getAdjmat(),teta,g->getSize());
    elapsed=cpuSecond()-start;
    std::cout<<"Lorena: sequential implementation ended in "<<elapsed<<" sec\n";

    start=cpuSecond();
    double *updated_teta=circleMap(g,teta);
    status_gpu=lorena_cpu::taglio_massimo(g->getAdjmat(),updated_teta,g->getSize());
    elapsed=cpuSecond()-start;
    std::cout<<"Lorena: parallel implementation ended in "<<elapsed<<" sec\n";


    if(check_output(status_cpu,status_gpu,g->getSize()))
        std::cout<<"------SUCCESS------\n";
    else
        std::cout<<"------ERROR: PARALLEL OUTPUT MUST AGREE WITH SEQUENTIAL ONE------\n";
    
    free(updated_teta);
    free(status_gpu);
    start=cpuSecond();
    updated_teta=circleMap(g,teta);
    status_gpu=maximumCut(g,updated_teta);
    elapsed=cpuSecond()-start;
    std::cout<<"Lorena: parallel implementation(version 2) ended in "<<elapsed<<" sec\n";


    if(check_output(status_cpu,status_gpu,g->getSize()))
        std::cout<<"------SUCCESS------\n";
    else
        std::cout<<"------ERROR: PARALLEL OUTPUT MUST AGREE WITH SEQUENTIAL ONE------\n";

    free(status_cpu);
    free(status_gpu);
    free(teta);
    delete g;

}
