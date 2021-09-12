#ifndef LORENA_CPU
#define LORENA_CPU
namespace lorena_cpu{
    void modifica_A_B(int *adjlist, double a, double b,double *A,double *B,int size);
    long taglio(int **adjmat,int *r,int size) ;
    int *taglio_massimo(int **adjmat,double *teta,int size);
    void mappa_cerchio_unitario(int **adjmat,double *teta,double *A,double *B,int size);
    int *mapAndCut(int **adjmat,double *teta,int size);
}
#endif
