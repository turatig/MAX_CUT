inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

inline void gpuErrCheck(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }
double cpuSecond();
