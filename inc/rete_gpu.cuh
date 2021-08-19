/*  -adjmat: adjacency matrix 
    -node: node index
    -status: vector representing the status of the network: status[i]=-1/1 -> node[i] belongs to partition -1/1
    -size: size of the graph
    -res: vector in which partial results of the sum reduction are stored
*/
#include "Graph.cuh"

void statusUpdate(int **adjmat,int node,int *status,int size,int *res);
/*  -g: Graph object representing the graph
*/
int *stabilizeHopfieldNet(Graph g);
