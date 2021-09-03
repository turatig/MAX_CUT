#ifndef LORENA_BATCH
#define LORENA_BATCH
__global__ void makePartitionsBatch(int *partitions,int oset,int batch_size,double *teta,int size);
__global__ void cutCostBatch(int *adjmat,int *partitions,int *res,int res_size,int batch_size,int size);
__global__ void maxInBatch(int *res,int res_size,int batch_size,int best_partition_idx,long best_partition_cost);
int *maximumCutBatch(Graph *g,double *teta,int batch_size);
#endif
