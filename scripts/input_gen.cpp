#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>

using namespace std;

int main(int argc, char **argv){
    ofstream args_file;
    int upper=75,lower=25;

    if(argc<2){
        std::cout<<"A file name must be given\n";
        return -1;
    }

    args_file.open(argv[1]);
    srand(time(NULL));

    for(int i=8;i<16;i++)
        args_file<<"Size: "<<(int)pow(2,i)<<" Seed: "<<rand()<<" Sparsity "<<(rand()%(upper-lower+1))+lower<<"\n";

    args_file<<"Size: "<<40064<<" Seed: "<<rand()<<" Sparsity "<<(rand()%(upper-lower+1))+lower<<"\n";
    args_file.close();
}
