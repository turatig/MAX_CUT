#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fstream>
#include "../inc/rete_cpu.cuh"
#include "../inc/rete_gpu.cuh"
#include "../inc/lorena_cpu.cuh"
#include "../inc/lorena_gpu.cuh"
#include "../inc/lorena_batch.cuh"
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
    long cost;
    int size, sparsity,seed;
    cudaDeviceProp dev_prop;

    cudaGetDeviceProperties(&dev_prop,0);
    std::cout<<"Zero-copy enabled: "<<dev_prop.canMapHostMemory<<"\n";
    
    std::ofstream hop_seq_out,hop_par_out,lor_seq_out,lor_par_out;

    hop_seq_out.open("hopfield_sequential.txt",std::ios_base::app);
    hop_par_out.open("hopfield_parallel.txt",std::ios_base::app);
    lor_par_out.open("lorena_parallel.txt",std::ios_base::app);
    
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

    hop_seq_out<<"Size: "<<size<<"\n";
    hop_par_out<<"Size: "<<size<<"\n";
    lor_seq_out<<"Size: "<<size<<"\n";
    lor_par_out<<"Size: "<<size<<"\n";

    Graph *g=new Graph(size,sparsity);
    start=cpuSecond();
    int * status_cpu=rete_cpu::stabilizza_rete_Hopfield(g->getAdjmat(),g->getSize());
    elapsed=cpuSecond()-start;
    std::cout<<"Hopfield: sequential implementation ended in "<<elapsed<<" sec\n";
    cost=lorena_cpu::taglio(g->getAdjmat(),status_cpu,g->getSize());
    std::cout<<"Hopfield->"<<cost<<"\n";
    hop_seq_out<<"Time: "<<elapsed<<" Cost: "<<cost<<"\n";
    
    start=cpuSecond();
    int *status_gpu=stabilizeHopfieldNet(g);
    elapsed=cpuSecond()-start;
    std::cout<<"Hopfield: parallel implementation ended in "<<elapsed<<" sec\n";
    cost=lorena_cpu::taglio(g->getAdjmat(),status_gpu,g->getSize());
    std::cout<<"Hopfield->"<<cost<<"\n";
    hop_par_out<<"Time: "<<elapsed<<" Cost: "<<cost<<"\n";

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
    lor_par_out<<"(Map)"<<" Time: "<<elapsed<<"\n";

    start=cpuSecond();
    status_gpu=maximumCut(g,updated_teta);
    elapsed=cpuSecond()-start;
    std::cout<<"Lorena--find best partition: parallel implementation ended in "<<elapsed<<" sec\n";
    //cost=lorena_cpu::taglio(g->getAdjmat(),status_gpu,g->getSize());
    lor_par_out<<"(Cut)"<<" Time: "<<elapsed<<" Cost: "<<cost<<"\n";


    if(check_output(status_cpu,status_gpu,g->getSize()) || lorena_cpu::taglio(g->getAdjmat(),status_cpu,g->getSize())==lorena_cpu::taglio(g->getAdjmat(),status_gpu,g->getSize()))
        std::cout<<"------SUCCESS------\n";
    else
        std::cout<<"------ERROR: PARALLEL OUTPUT MUST AGREE WITH SEQUENTIAL ONE------\n";

    start=cpuSecond();
    status_gpu=maximumCutBatch(g,updated_teta,256);
    elapsed=cpuSecond()-start;
    std::cout<<"Lorena--find best partition: parallel batch implementation ended in "<<elapsed<<" sec\n";
    std::cout<<"Partition's cost :"<<lorena_cpu::taglio(g->getAdjmat(),status_gpu,g->getSize())<<"\n";
    lor_par_out<<"(Cut_Batch)"<<" Time: "<<elapsed<<"\n";

    if(check_output(status_cpu,status_gpu,g->getSize()) || lorena_cpu::taglio(g->getAdjmat(),status_cpu,g->getSize())==lorena_cpu::taglio(g->getAdjmat(),status_gpu,g->getSize()))
        std::cout<<"------SUCCESS------\n";
    else
        std::cout<<"------ERROR: PARALLEL OUTPUT MUST AGREE WITH SEQUENTIAL ONE------\n";
    
    
    hop_seq_out.close();
    hop_par_out.close();
    lor_par_out.close();

    free(status_cpu);
    free(status_gpu);
    free(teta);
    free(updated_teta);
    delete g;

}
