#ifndef UTILS
#define UTILS
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define gpuErrCheck( ans) { gpuAssert((ans), __FILE__, __LINE__); }
double cpuSecond();
/*
Function used to check the correctness of the parallel algorithm with respect to its sequential implementation
*/
bool check_output(int *benchmark,int *tested,int size);
void printPartitions(int *partitions,int size);
#endif
