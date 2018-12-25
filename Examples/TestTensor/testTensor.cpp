//
// Created by Hui Xie on 7/17/2018.
//
#include <iostream>
#include "CDLF.h"
#include <list>



int main (int argc, char *argv[]) {

    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    cout<<"Info: program use Cuda GPU."<<endl;
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
#else

    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif
    printCurrentLocalTime();

    Tensor<float> tensor1({3,4});
    int k=0;
    for (int i=0; i<3; ++i){
        for(int j=0; j<4;++j){
            tensor1.e(k) = k;
            ++k;
        }
    }
    cout <<"tensor1:"<<endl;
    tensor1.print();

    Tensor<float> tensorTranspose = tensor1.transpose();

    cout<<"Transpose: "<<endl;
    tensorTranspose.print();

    Tensor<float> tensor2({4,2});
    k =0;
    for (int i=0; i<4; ++i){
        for(int j=0; j<2;++j){
            tensor2.e(k) = 1;
            ++k;
        }
    }
    cout <<"tensor2:"<<endl;
    tensor2.print();

    Tensor<float> tensor3 = tensor1* tensor2;
    cout <<"tensor3 = tensor1* tensor2 = "<<endl;
    tensor3.print();

    Tensor<float> tensor4({3,4});
    k =0;
    for (int i=0; i<3; ++i){
        for(int j=0; j<4;++j){
            tensor4.e(k) = 1;
            ++k;
        }
    }
    cout <<"tensor4:"<<endl;
    tensor4.print();

    Tensor<float> tensor5 = tensor1+  tensor4;
    cout <<"tensor5 = tensor1+  tensor4 = "<<endl;
    tensor5.print();

    Tensor<float> tensor6 = tensor1-  tensor4;
    cout <<"tensor6 = tensor1-  tensor4 = "<<endl;
    tensor6.print();

    Tensor<float> tensor7 = tensor6/2;
    cout <<"tensor7 = tensor6/2 = "<<endl;
    tensor7.print();

    tensor7 += 2;
    cout <<"tensor7 += 2 = "<<endl;
    tensor7.print();

    tensor7 -= 2;
    cout <<"tensor7 -= 2 = "<<endl;
    tensor7.print();

    tensor7 *= 2;
    cout <<"tensor7 *= 2 = "<<endl;
    tensor7.print();

    tensor7 /= 2;
    cout <<"tensor7 /= 2 = "<<endl;
    tensor7.print();

    Tensor<float> tensor8 = tensor7 +2;
    cout <<"tensor8 = tensor7 +2 = "<<endl;
    tensor8.print();

    Tensor<float> tensor9 = tensor8- 2;
    cout <<" tensor9 = tensor8- 2; = "<<endl;
    tensor9.print();

    Tensor<float> tensor10 = tensor9* 2;
    cout <<" tensor10 = tensor9* 2 = "<<endl;
    tensor10.print();

    Tensor<float> tensor11 = tensor10 / 2;
    cout <<" tensor11 = tensor10/ 2 = "<<endl;
    tensor11.print();

    cout <<"tensor1:"<<endl;
    tensor1.print();

    cout <<"tensor4:"<<endl;
    tensor4.print();

    tensor1 -= tensor4;
    cout <<" tensor1 -= tensor4; = "<<endl;
    tensor1.print();

    tensor1 += tensor4;
    cout <<" tensor1 += tensor4; = "<<endl;
    tensor1.print();

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
    tensor1.print();

    cout<<"Test operator + "<<endl;
    vector<int> aVector={1,2,3,4,5};
    cout<<"Original Vector:";
    printVector(aVector);
    vector<int> bVector = aVector+2;
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
    tensor20.print();

    cout <<"tensor20:"<<endl;
    tensor20.print();
    Tensor<float>* pTensor22 = nullptr;
    tensor20.extractLowerDTensor(3, pTensor22);
    cout<<" tensor22: by tensor20.extractLowerDTensor(3, pTensor22):"<<endl;
    pTensor22->print();
    if (nullptr != pTensor22){
        delete pTensor22;
    }

    cout <<"tensor20:"<<endl;
    tensor20.print();

    Tensor<float>* pTensor23 = new Tensor<float>({3,3});
    tensor20.subTensorFromTopLeft({0,0},pTensor23,2);
    cout<<"tensor23: by te tensor20.subTensorFromTopLeft({0,0},{3,3},pTensor23,2):"<<endl;
    pTensor23->print();

    cout <<"tensor20:"<<endl;
    tensor20.print();
    tensor20.subTensorFromTopLeft(1,pTensor23,2);
    cout<<"tensor23: by te tensor20.subTensorFromTopLeft(1,pTensor23,2):"<<endl;
    pTensor23->print();



    if (nullptr != pTensor23){
        delete pTensor23;
    }

    cout <<"tensor20:"<<endl;
    tensor20.print();

    cout<<"row(4) = tensor20.row(4)="<<endl;
    tensor20.row(4).print();

    cout<<"colunmn(3) = tensor20.column(3)="<<endl;
    tensor20.column(3).print();

    cout<<"Test getMaxPositionSubTensor:"<<endl;
    Tensor<float> tensor30({7,5});
    generateGaussian(&tensor30, 2,3);
    cout<<"Tensor30 :"<<endl;
    tensor30.print();
    cout<<"subTensorMaxPos = tensor30.getMaxPositionSubTensor(): "<<endl;
    Tensor<unsigned char> subTensorMaxPos = tensor30.getMaxPositionSubTensor();
    subTensorMaxPos.print();

    cout<<"========End of Test Basic Tensor=========="<<endl;

    cout<<"=========Start Test MKL CBLAS Tensor========="<<endl;

    Tensor<float> A= tensor1;
    cout<<"A:"<<endl;
    A.print();  //{3*4} matrix

    Tensor<float> x({4,1});
    for (int i=0; i<4; ++i){
        x.e(i) = 1;
    }
    cout<<"x:"<<endl;
    x.print();

    Tensor<float> b({3,1});
    for (int i=0; i<3; ++i){
        b.e(i) = i;
    }
    cout<<"b:"<<endl;
    b.print();


    Tensor<float> y({3,1});
    cout<<"y = Ax+b:"<<endl;
    gemv(false, &A, &x, &b, &y);
    cout<<"y:"<<endl;
    y.print();

    cout<<"now b = "<<endl;
    b.print();

    cout<<"b = Ax+b:"<<endl;
    gemv(false, &A, &x, &b);
    b.print();


    cout<<"y=3.0*b +y:"<<endl;
    axpy(3,&b,&y);
    y.print();


    Tensor<float> B({4,3});
    k=0;
    for (int i=0; i<4; ++i){
        for(int j=0; j<3;++j){
            B.e(k) = k*2;
            ++k;
        }
    }
    cout <<"B:"<<endl;
    B.print();

    Tensor<float> C({3,3});
    for (int i=0; i< C.getLength(); ++i){
        C.e(i) = 1;
    }
    cout <<"C:"<<endl;
    C.print();
    // C = a*A*B+ b*C
    gemm(3, false, &A, false, &B, 2, &C);
    cout<<"C= 3*A*B+ 2*C: "<<endl;
    C.print();
    
    Tensor<float> Bt = B.transpose();
    cout <<"Bt: "<<endl;
    Bt.print();
    
    Tensor<float> D({3,4});

    matAdd(3, &A, 4, &Bt, &D);
    cout<<"D = 3*A+ 4*Bt= "<<endl;
    D.print();


    cout<<"=========End of Test MKL CBLAS Tensor========="<<endl;




}


