#include<iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "utils.h"
#include "Graph.h"

/*
Create undirected graph, with a fraction of p nodes connected:
    -allocates memory both on cpu and gpu
*/
Graph::Graph(int s,int p){
    size=s;

    adjmat=(int**)malloc(size*sizeof(int*));
    
    //srand(time(NULL));
    for(int i=0;i<size;i++){

        adjmat[i]=(int*)malloc(size*sizeof(int));

        for(int j=0;j<size;j++){

            //Graph is supposed to be undirected, so adjacency matrix is symmetric
            if(j<i){ adjmat[i][j]=adjmat[j][i]; }

            else{

                if(i!=j&&rand()%100<p){  adjmat[i][j]=1; }
                else{ adjmat[i][j]=0; }

            }
        }
    }
}

/*
Create graph object from input file
*/
Graph::Graph(char *argv) {
    char s[11];
    FILE *in;

    if( (in = fopen(argv,"r")) == NULL ) {
      printf("File non trovato!\n");
      exit(1);
    }

    size = atoi( fgets(s,10,in));
    std::cout<<"CREATING GRAPH OF SIZE "<<size<<"\n";
    adjmat=(int**)malloc(size*sizeof(int *));
    
   
    for (int i=0; i<size; i++) {
        adjmat[i]=(int*)malloc(size*sizeof(int));
        for(int j=0; j<size; j++)
          adjmat[i][j] = fgetc(in)-48;

        fgetc(in);
      
    }
    fclose(in);
}

Graph::~Graph(){
    for(int i=0;i<size;i++)
        free(adjmat[i]);
    free(adjmat);
}

int ** Graph::getAdjmat(){ return adjmat;}
int Graph::getSize(){ return size;}
void Graph::print(){
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++)
            std::cout<<adjmat[i][j]<<" ";
        std::cout<<"\n";
    }
}
