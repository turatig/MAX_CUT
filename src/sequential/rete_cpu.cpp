/**************************************************************

Algoritmo approssimato per la ricerca della cricca max di un
grafo basato su di una successione di reti di Hopfield discrete.
  
  Esecuzione:
        
	 % run <graph_name> 
 dove
		- graph_name = file contenente la matrice di incidenza.

 **************************************************************/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rete_cpu.h"
#include "utils.h"

/****************************************************************/

void rete_cpu::energia(int **adjmat,int *stato_rete,int size)
{
  register int i,j;
  int s=0;

  for (i=0; i<size-1; i++)
    for (j=i+1; j<size; j++)
      s += adjmat[i][j]*(1 - stato_rete[i]*stato_rete[j]); 
  printf("\n Energia -> %d\n",s/2);
}

/****************************************************************/

int *rete_cpu::stabilizza_rete_Hopfield(int **adjmat,int size) {
  register int i,j;
  int pred, FINE=1, somme;
  int *stato_rete=(int*)calloc(size,sizeof(int));
    
  while (FINE) {
    FINE = 0;
    for (i=0; i<size; i++) {
      somme = 0;
      // somma pesata dei vicini
      for (j=0; j<size; j++)
        somme -= adjmat[i][j]*stato_rete[j];
    


      // analisi dello stato
      pred = stato_rete[i];
      if (somme<0)    
        stato_rete[i] = -1;
      else            
        stato_rete[i] = 1;

      // test cambio stato
      if (pred != stato_rete[i]) 
        FINE = 1;   
    }
  }
  return stato_rete;
}

/******************************    Main     ******************************/

/*int main(int argc, char *argv[]) {
  register int k;

  if( argc < 2 ) {
    printf("Manca un parametro!\n");
    exit(1);
  }
  inizializza_strutture(argv[1]);
  stabilizza_rete_Hopfield();
  energia();
  // stampa_stato_rete();
  stampa_stato_rete();
  return(0);
}*/
