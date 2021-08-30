#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../inc/lorena_cpu.cuh"


#define Min(x,y) (fabs(x) < fabs(y) ? fabs(x) : fabs(y))

#define EPSILON 5
#define PI 3.14159265358979323846

/****************************************************************/

void lorena_cpu::modifica_A_B(int *adjlist, double a, double b,double *A,double *B,int size) {
	for (int i=0; i<size; i++) {
		A[i] += (double) a*adjlist[i];
		B[i] += (double) b*adjlist[i];
	}
}

/****************************************************************/

double lorena_cpu::taglio(int **adjmat,int *r,int size) {
	register int i,j;
	double s=0;

	for (i=0; i<size-1; i++){
		for (j=i+1; j<size; j++){
			s += adjmat[i][j]*(1-r[i]*r[j]);
        }
    }

	return s/2;
}

/****************************************************************/

int *lorena_cpu::taglio_massimo(int **adjmat,double *teta,int size) {
	register int i,j,k;
	register double t,T=0;
	int *R;
    double max_alfa;
	double alfa;

	R = (int *) calloc(size,sizeof(int));
    int *res=(int*)malloc(size*sizeof(int));

	for (i=0; i<size; i++) {
		alfa = (teta[i] > PI) ? teta[i]-PI : teta[i];
		for (j=0; j<size; j++)
			R[j] = ((teta[j] >= alfa) && (teta[j] < alfa+PI)) ? 1 : -1;
		t = taglio(adjmat,R,size);
		if ( t > T ){
            T = t;
            max_alfa=alfa;
        }
            
	}
    
    /*Store the resulted partition and return the pointer*/
    for(j=0;j<size;j++)
        res[j] = ((teta[j] >= max_alfa) && (teta[j] < max_alfa+PI)) ? 1 : -1;
	std::cout<<"Lorena ->"<<T<<"\n";
    free(R);
    return res;
}

void lorena_cpu::mappa_cerchio_unitario(int **adjmat,double *teta,double *A,double *B,int size){
	int OK = 1, nround = 0;
	double p,alfa;
    
	while ( OK ) {
		nround++;
		OK = 0;
		for (int k=0; k<size; k++) {
			alfa = teta[k];
			teta[k] = atan(B[k]/A[k]); 
			if (A[k] >= 0) 
				teta[k] += PI; 
			else if (B[k] > 0) 
				teta[k] += 2*PI;
			modifica_A_B(adjmat[k], cos(teta[k])-cos(alfa), sin(teta[k])-sin(alfa),A,B,size);
			if ( Min(alfa-teta[k],2*PI-alfa+teta[k]) > EPSILON )
				OK = 1;
		}
	}
}

int *lorena_cpu::mapAndCut(int **adjmat,double *teta,int size){
    double *t=(double*)malloc(size*sizeof(double));
    /*Hard copy value of teta to avoid external changes*/
    memcpy(t,teta,size*sizeof(double));

    double *A=(double*)malloc(size*sizeof(double));
    double *B=(double*)malloc(size*sizeof(double));
    
	for (int i=0; i<size; i++) {
		A[i] = B[i] = 0;
		for(int j=0; j<size; j++) {
			A[i] += adjmat[i][j]*cos(teta[j]);
			B[i] += adjmat[i][j]*sin(teta[j]);
		}
	}
    
    lorena_cpu::mappa_cerchio_unitario(adjmat,t,A,B,size);
    //std::cout<<"Start cutting the mapped points\n";
    int *res=lorena_cpu::taglio_massimo(adjmat,t,size);
    free(A);
    free(B);
    return res;
}

/****************************************************************/

/*void lorena_cpu::inizializza_strutture(int argc, char *argv[]){
	register int i,j;
	char s[11];

	if ( argc < 3 ) {
		printf("Manca un parametro!\n");
		exit(1);
	}
	if ((in = fopen(argv[1],"r")) == NULL) {
		printf("File non trovato!\n");
		exit(1);
	}
	srand(atoi(argv[2]));  // seme fissato x ripetere stesso exp
	
	// srand(time(NULL)&RAND_MAX);
	N = atoi(fgets(s,10,in));
	adjmat = (Row *) malloc(N*sizeof(Row));
	teta = (double *) calloc(N,sizeof(double));
	A = (double *) calloc(N,sizeof(double));
	B = (double *) calloc(N,sizeof(double));
	for (i=0; i<N; i++)
		teta[i] = (double) 2*PI*rand()/RAND_MAX;
	for (i=0; i<N; i++) {
		adjmat[i] = (int *) malloc(N*sizeof(int));
		A[i] = B[i] = 0;
		for(j=0; j<N; j++) {
			adjmat[i][j] = fgetc(in)-48;
			A[i] += adjmat[i][j]*cos(teta[j]);
			B[i] += adjmat[i][j]*sin(teta[j]);
		}
		fgetc(in);
	}
	fclose(in);
}*/

/*Init with externally created input*/
/*void lorena_cpu::inizializza_strutture(int **adjmat,int size){
    N=size;
    //adjmat is supposed to be read-only input, so pointer can be assigned
    adjmat=adjmat;
    teta=(double*)malloc(size*sizeof(double));
    A=(double*)malloc(size*sizeof(double));
    B=(double*)malloc(size*sizeof(double));
    
    //vector teta,A,B are modified during the algorithm, so they must be hard-copied

	for (int i=0; i<N; i++)
		teta[i] = (double) 2*PI*rand()/RAND_MAX;
	for (int i=0; i<N; i++) {
		A[i] = B[i] = 0;
		for(int j=0; j<N; j++) {
			A[i] += adjmat[i][j]*cos(teta[j]);
			B[i] += adjmat[i][j]*sin(teta[j]);
            std::cout<<"Init lorena\n";
		}
	}
    std::cout<<"Init lorena\n";
}*/
/**************************************************************/

/******************************    Main     ******************************/

/*int main(int argc, char *argv[]) {
	register int k;
	int OK = 1, nround = 0;
	double p;

	inizializza_strutture(argc,argv);
	while ( OK ) {
		nround++;
		OK = 0;
		for (k=0; k<N; k++) {
			alfa = teta[k];
			teta[k] = atan(B[k]/A[k]); 
			if (A[k] >= 0) 
				teta[k] += PI; 
			else if (B[k] > 0) 
				teta[k] += 2*PI;
			modifica_A_B(k, cos(teta[k])-cos(alfa), sin(teta[k])-sin(alfa));
			if ( Min(alfa-teta[k],2*PI-alfa+teta[k]) > EPSILON )
				OK = 1;
		}
	}
	taglio_massimo();
	stampa_teta();
	printf("Num rounds = %d\n",nround);
	return(0);
}*/




