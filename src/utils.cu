#include <stdio.h>
#include <sys/time.h>
#include "../inc/utils.cuh"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
