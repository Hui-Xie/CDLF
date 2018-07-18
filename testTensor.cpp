//
// Created by Sheen156 on 7/17/2018.
//
#include <iostream>
#include "Tensor.h"
#include <list>


int main (int argc, char *argv[]) {
    Tensor<float> tensor1({3,4});
    int k=0;
    for (int i=0; i<3; ++i){
        for(int j=0; j<4;++j){
            tensor1.e(k) = k;
            ++k;
        }
    }
    cout <<"tensor1:"<<endl;
    tensor1.printElements();

    Tensor<float> tensorTranspose = tensor1.transpose();

    cout<<"Transpose: "<<endl;
    tensorTranspose.printElements();

    Tensor<float> tensor2({4,2});
    k =0;
    for (int i=0; i<4; ++i){
        for(int j=0; j<2;++j){
            tensor2.e(k) = 1;
            ++k;
        }
    }
    cout <<"tensor2:"<<endl;
    tensor2.printElements();

    Tensor<float> tensor3 = tensor1* tensor2;
    cout <<"tensor3 = tensor1* tensor2 = "<<endl;
    tensor3.printElements();

    Tensor<float> tensor4({3,4});
    k =0;
    for (int i=0; i<3; ++i){
        for(int j=0; j<4;++j){
            tensor4.e(k) = 1;
            ++k;
        }
    }
    cout <<"tensor4:"<<endl;
    tensor4.printElements();

    Tensor<float> tensor5 = tensor1+  tensor4;
    cout <<"tensor5 = tensor1+  tensor4 = "<<endl;
    tensor5.printElements();

    Tensor<float> tensor6 = tensor1-  tensor4;
    cout <<"tensor6 = tensor1-  tensor4; = "<<endl;
    tensor6.printElements();

    Tensor<float> tensor7 = tensor6/2;
    cout <<"tensor7 = tensor6/2 = "<<endl;
    tensor7.printElements();

    tensor7 += 2;
    cout <<"tensor7 += 2; = "<<endl;
    tensor7.printElements();

    tensor7 -= 2;
    cout <<"tensor7 -= 2; = "<<endl;
    tensor7.printElements();

    tensor7 *= 2;
    cout <<"tensor7 *= 2; = "<<endl;
    tensor7.printElements();

    tensor7 /= 2;
    cout <<"tensor7 /= 2; = "<<endl;
    tensor7.printElements();

    Tensor<float> tensor8 = tensor7 +2;
    cout <<"tensor8 = tensor7 +2; = "<<endl;
    tensor8.printElements();

    Tensor<float> tensor9 = tensor8- 2;
    cout <<" tensor9 = tensor8- 2; = "<<endl;
    tensor9.printElements();

    Tensor<float> tensor10 = tensor9* 2;
    cout <<" tensor10 = tensor9* 2 = "<<endl;
    tensor10.printElements();

    Tensor<float> tensor11 = tensor10/ 2;
    cout <<" tensor11 = tensor10/ 2; = "<<endl;
    tensor11.printElements();

    cout <<"tensor1:"<<endl;
    tensor1.printElements();

    cout <<"tensor4:"<<endl;
    tensor4.printElements();

    tensor1 -= tensor4;
    cout <<" tensor1 -= tensor4; = "<<endl;
    tensor1.printElements();

    tensor1 += tensor4;
    cout <<" tensor1 += tensor4; = "<<endl;
    tensor1.printElements();

    cout<<"Test [] overload"<<endl;
    cout<<"tensor1[3]="<<tensor1[3]<<endl;
    cout<<"tensor1(3)="<<tensor1(3)<<endl;
    cout<<"tensor1(0,2)="<<tensor1(0,2)<<endl;
    cout<<"tensor1(1,2)="<<tensor1(1,2)<<endl;
    cout<<"tensor1(2,1)="<<tensor1(2,1)<<endl;
    cout<<"tensor1(2,1)= tensor(2,1)+3"<<endl;
    tensor1(2,1) += 3;
    cout<<"tensor1(2,1)="<<tensor1(2,1)<<endl;
    tensor1.printElements();

}


