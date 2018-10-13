//
// Created by Hui Xie on 7/17/2018.
//
#include <iostream>
#include "CDLF.h"
#include <list>
#include "GPUAttr.h"


int main (int argc, char *argv[]) {

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

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
    cout <<"tensor6 = tensor1-  tensor4 = "<<endl;
    tensor6.printElements();

    Tensor<float> tensor7 = tensor6/2;
    cout <<"tensor7 = tensor6/2 = "<<endl;
    tensor7.printElements();

    tensor7 += 2;
    cout <<"tensor7 += 2 = "<<endl;
    tensor7.printElements();

    tensor7 -= 2;
    cout <<"tensor7 -= 2 = "<<endl;
    tensor7.printElements();

    tensor7 *= 2;
    cout <<"tensor7 *= 2 = "<<endl;
    tensor7.printElements();

    tensor7 /= 2;
    cout <<"tensor7 /= 2 = "<<endl;
    tensor7.printElements();

    Tensor<float> tensor8 = tensor7 +2;
    cout <<"tensor8 = tensor7 +2 = "<<endl;
    tensor8.printElements();

    Tensor<float> tensor9 = tensor8- 2;
    cout <<" tensor9 = tensor8- 2; = "<<endl;
    tensor9.printElements();

    Tensor<float> tensor10 = tensor9* 2;
    cout <<" tensor10 = tensor9* 2 = "<<endl;
    tensor10.printElements();

    Tensor<float> tensor11 = tensor10 / 2;
    cout <<" tensor11 = tensor10/ 2 = "<<endl;
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
    cout<<"tensor1.e({2,2}) = "<< tensor1.e({2,2})<<endl;
    cout<<"tensor1(0,2)="<<tensor1(0,2)<<endl;
    cout<<"tensor1(1,2)="<<tensor1(1,2)<<endl;
    cout<<"tensor1(2,1)="<<tensor1(2,1)<<endl;
    cout<<"tensor1(2,1)= tensor(2,1)+3"<<endl;
    tensor1(2,1) += 3;
    cout<<"tensor1(2,1)="<<tensor1(2,1)<<endl;
    tensor1.printElements();

    cout<<"Test operator + "<<endl;
    vector<long> aVector={1,2,3,4,5};
    cout<<"Original Vector:";
    printVector(aVector);
    vector<long> bVector = aVector+2;
    cout<<" bVector = aVector+2; "<<endl;
    printVector(bVector);

    cout<<"bVecotr*3: "<<endl;
    printVector(bVector*3);

    cout<<"test subTensorFromCenter"<<endl;
    Tensor<float> tensor20({10,10});
    k=0;
    for (int i=0; i<10; ++i){
        for(int j=0; j<10;++j){
            tensor20.e(k) = k;
            ++k;
        }
    }
    cout <<"tensor20:"<<endl;
    tensor20.printElements();

    cout<<"tensor21 = tensor20.subTensorFromCenter({2,3}, {5,5});"<<endl;
    Tensor<float>* pTensor21 = nullptr;
    tensor20.subTensorFromCenter({2,3}, {5,5}, pTensor21);
    pTensor21->printElements();
    if (nullptr != pTensor21){
        delete pTensor21;
    }


    cout <<"tensor20:"<<endl;
    tensor20.printElements();
    Tensor<float>* pTensor22 = nullptr;
    tensor20.extractLowerDTensor(3, pTensor22);
    cout<<" tensor22: by tensor20.extractLowerDTensor(3, pTensor22):"<<endl;
    pTensor22->printElements();
    if (nullptr != pTensor22){
        delete pTensor22;
    }

    cout <<"tensor20:"<<endl;
    tensor20.printElements();

    Tensor<float>* pTensor23;
    tensor20.subTensorFromTopLeft({0,0},{3,3},pTensor23,2);
    cout<<"tensor23: by te tensor20.subTensorFromTopLeft({0,0},{3,3},pTensor23,2):"<<endl;
    pTensor23->printElements();
    if (nullptr != pTensor23){
        delete pTensor23;
    }

    cout <<"tensor20:"<<endl;
    tensor20.printElements();

    cout<<"row(4) = tensor20.row(4)="<<endl;
    tensor20.row(4).printElements();

    cout<<"colunmn(3) = tensor20.column(3)="<<endl;
    tensor20.column(3).printElements();

    cout<<"Test getMaxPositionSubTensor:"<<endl;
    Tensor<float> tensor30({7,5});
    generateGaussian(&tensor30, 2,3);
    cout<<"Tensor30 :"<<endl;
    tensor30.printElements();
    cout<<"subTensorMaxPos = tensor30.getMaxPositionSubTensor(): "<<endl;
    Tensor<unsigned char> subTensorMaxPos = tensor30.getMaxPositionSubTensor();
    subTensorMaxPos.printElements();

    cout<<"========End of Test Tensor=========="<<endl;






}


