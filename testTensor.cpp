//
// Created by Sheen156 on 7/17/2018.
//
#include <iostream>
#include "Tensor.h"


int main (int argc, char *argv[]) {
    Tensor<float> tensor({3,4});
    int k=0;
    for (int i=0; i<3; ++i){
        for(int j=0; j<4;++j){
            tensor.e(k) = k;
            ++k;
        }
    }
    tensor.printElements();

    Tensor<float> tensorTranspose = tensor.transpose();

    cout<<"Transpose: "<<endl;
    tensorTranspose.printElements();



}


