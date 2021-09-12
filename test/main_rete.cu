#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include "../inc/rete_cpu.cuh"
#include "../inc/Graph.cuh"
#include "../inc/utils.cuh"

int main(){
    
    Graph *g=new Graph((int)pow(2,14),35);
    double start=cpuSecond();
    int *status_gpu=stabilizeHopfieldNet(g);
    double elapsed=cpuSecond()-start;
    std::cout<<"Hopfield: parallel implementation ended in "<<elapsed<<" sec\n";
    long cost=lorena_cpu::taglio(g->getAdjmat(),status_gpu,g->getSize());
    std::cout<<"Hopfield->"<<cost<<"\n";
    hop_par_out<<"Time: "<<elapsed<<" Cost: "<<cost<<"\n";
}
