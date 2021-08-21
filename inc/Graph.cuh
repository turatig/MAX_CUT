/*
Class to manage a random generated undirected graph.
*/
#ifndef GRAPH
#define GRAPH
class Graph{
    private:
        /*  -adjmat: adjacency matrix. Host memory pointer.
            -gpu_adjmat: adjacency matrix. Device memory pointer. In linearized form.
            -size: number of nodes in the graph.
        */
        int **adjmat;
        int *gpu_adjmat;
        int size;
    public:
        /*  -s: size.
            -p: fraction of connected nodes.
        */
        Graph(int s,int p=75);
        /*Init from file*/
        Graph(char *argv);
        ~Graph();
        int **getAdjmat();
        int *getDevicePointer();
        int getSize();
        void print();
};
#endif
