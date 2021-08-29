#ifndef RETE_CPU
#define RETE_CPU
namespace rete_cpu{
    void energia(int **adjmat,int *stato_rete,int size);
    int *stabilizza_rete_Hopfield(int **adjmat,int size);
}
#endif
