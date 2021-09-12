#ifndef UTILS
#define UTILS
double cpuSecond();
/*
Function used to check the correctness of the parallel algorithm with respect to its sequential implementation
*/
bool check_output(int *benchmark,int *tested,int size);
void printPartitions(int *partitions,int size);
#endif
