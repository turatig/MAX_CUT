#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fstream>
#include "rete_cpu.h"
#include "lorena_cpu.h"
#include "Graph.h"
#include "utils.h"


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
    long cost;
    int size, sparsity,seed;
    
    std::ofstream hop_seq_out;

    hop_seq_out.open("hopfield_sequential.txt",std::ios_base::app);
  
    if(argc<2){
        std::cout<<"The size of the graph must be specified\n";
        return -1;
    }
    size=atoi(argv[1]);
    if(argc>=3){
        seed=atoi(argv[2]);
        srand(seed);
    }
    else{
        std::cout<<"Random seeded\n";
        seed=time(NULL);
        srand(seed);
    }
    if(argc>=4)
        sparsity=atoi(argv[3]);
    else
        sparsity=rand()%100;

    std::cout<<"Size: "<<size<<" Seed: "<<seed<<" Sparsity: "<<sparsity<<"\n";

    hop_seq_out<<"Size: "<<size<<"\n";

    Graph *g=new Graph(size,sparsity);
    start=cpuSecond();
    int * status_cpu=rete_cpu::stabilizza_rete_Hopfield(g->getAdjmat(),g->getSize());
    elapsed=cpuSecond()-start;
    std::cout<<"Hopfield: sequential implementation ended in "<<elapsed<<" sec\n";
    cost=lorena_cpu::taglio(g->getAdjmat(),status_cpu,g->getSize());
    std::cout<<"Hopfield->"<<cost<<"\n";
    hop_seq_out<<"Time: "<<elapsed<<" Cost: "<<cost<<"\n";
    
    free(status_cpu);
    
    double *teta=(double*)malloc(g->getSize()*sizeof(double));
	for (int i=0; i<g->getSize(); i++)
		teta[i] = (double) 2*PI*rand()/RAND_MAX;
    
    //Execution time is logged in the function
    status_cpu=lorena_cpu::mapAndCut(g->getAdjmat(),teta,g->getSize());

    hop_seq_out.close();

    free(status_cpu);
    free(teta);
    delete g;

}
