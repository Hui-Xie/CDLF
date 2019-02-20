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
    tensor20.subTensorFromTopLeft({0,0},pTensor23,{2,2});
    cout<<"tensor23: by tensor20.subTensorFromTopLeft({0,0}, pTensor23,{2,2}):"<<endl;
    pTensor23->print();


    cout <<"tensor20:"<<endl;
    tensor20.print();
    Tensor<float>* pTensor23A = new Tensor<float>({2,3});
    tensor20.subTensorFromTopLeft({0,0},pTensor23A,{1,1});
    cout<<"tensor23A: by tensor20.subTensorFromTopLeft({0,0},pTensor23A,{1,1}):"<<endl;
    pTensor23A->print();

    if (nullptr != pTensor23A){
        delete pTensor23A;
    }

    cout <<"tensor20:"<<endl;
    tensor20.print();
    tensor20.subTensorFromTopLeft(1,pTensor23,{2,2});
    cout<<"tensor23: by te tensor20.subTensorFromTopLeft(1,pTensor23,2):"<<endl;
    pTensor23->print();

    if (nullptr != pTensor23){
        delete pTensor23;
    }

    cout <<"Test 3D Tensor{6,8,9} with element orderly increasging by 1"<<endl;
    Tensor<float> matrix3D({6,8,9});
    for(int i=0; i<matrix3D.getLength(); ++i){
        matrix3D.e(i) = i;
    }
    Tensor<float> subMatrix3D({2,3,3});
    matrix3D.subTensorFromTopLeft(0, &subMatrix3D, {1,1,1});
    for (int i=0; i<2;++i){
        printf("the %d slice of subMatrix3D(2,3,3)\n", i);
        subMatrix3D.slice(i).print();
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

    //read a Tensor file and output its sigma.
    /*
    Tensor<float> tensor120({90,500,500});
    string  filename = "/Users/hxie1/temp_netParameters/HNSCC_matrix/groundTruth.csv";
    tensor120.load(filename);
    float mean = tensor120.average();
    float sigma = sqrt(tensor120.variance());
    const int N = tensor120.getLength();
    int count1 =0;
    for (int i=0; i< N; ++i){
        if (1 == tensor120.e(i)) ++count1;
    }


    cout<<filename<<endl;
    printf("mean = %f, sigma = %f\n", mean, sigma);
    printf("count1 = %d \n", count1);
    */
    
    
    cout<<"Test get center of nonZero elements of a Tensor"<<endl;
    Tensor<float> tensor130({15,20});
    tensor130.zeroInitialize();
    tensor130.e(45) = 1;
    tensor130.e(46) = 1;
    tensor130.e(47) = 1;
    tensor130.e(65) = 1;
    tensor130.e(66) = 1;
    tensor130.e(67) = 1;
    tensor130.e(85) = 1;
    tensor130.e(86) = 1;
    tensor130.e(87) = 1;
    vector<int> center = tensor130.getCenterOfNonZeroElements();
    cout<<"tensor 130, size: 15*20"<<endl;
    tensor130.print(true);
    cout<<"center: "<<vector2Str(center)<<endl;


    cout<<"Test generatePositiveNegativeRandom"<<endl;

    vector<int> drift = generatePositiveNegativeRandomNumber(5, 9);

    cout<<"generatePositiveNegativeRandom: "<< vector2Str(drift)<<endl;


    //============================
    //Test rotation 3D tensor

    cout<<"==========Test rotation 3D tensor =================== "<<endl;
    Tensor<float> tensor140({3,4,5});
    const int N140 = tensor140.getLength();
    for(int i=0; i<N140; ++i){
        tensor140.e(i) = i;
    }

    cout<<"Original tensor size: "<<vector2Str(tensor140.getDims())<<endl;
    cout<<"before rotation: slice(1):"<<endl;
    tensor140.slice(1).print();

    vector<float> radianVec = generatePositiveNegativeRandomRadian(3,M_PI/4.0);

    Tensor<float>* pRotatedTensor = nullptr;
    tensor140.rotate3D(radianVec, IPPI_INTER_NN, pRotatedTensor);
    cout<<"roatated tensor size: "<<vector2Str(pRotatedTensor->getDims())<<endl;
    cout<<"After rotation: slice(1):"<<endl;
    pRotatedTensor->slice(1).print();

    delete pRotatedTensor;
    

}


