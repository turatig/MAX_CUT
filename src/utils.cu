#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include "../inc/utils.cuh"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

bool check_output(int *benchmark,int *tested,int size){
    for(int i=0;i<size;i++){
        if(benchmark[i]!=tested[i]){
            std::cout<<"Mismatch at idx "<<i<<" "<<benchmark[i]<<" "<<tested[i]<<"\n";
            return false;
        }
    }
    return true;
}

void printPartitions(int *partitions,int size){
    std::cout<<"Graph partitions\n";

    for(int i=0;i<size;i++){
        std::cout<< (int)(partitions[i]+1)/2<<" ";
    }
    std::cout<<"\n";
}
