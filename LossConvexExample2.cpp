//
// Created by Hui Xie on 7/24/2018.
//

#include "LossConvexExample2.h"
#include <iostream>
#include <math.h>       /* pow */
using namespace std;

// L(x) = \sum exp(x_i -i)

LossConvexExample2::LossConvexExample2(const int id, const string& name): LossLayer(id,name){

}
LossConvexExample2::~LossConvexExample2(){

}


float LossConvexExample2::lossCompute(){
    //use m_prevLayerPointer->m_pYTensor,
    m_loss = 0;
    Tensor<float> & prevY = *(m_prevLayer->m_pYTensor);
    long N = prevY.getLength();
    for (long i=0; i< N ;++i){
        m_loss += exp( prevY.e(i) - i);
    }
    return m_loss;
}

// f(x) = \sum exp(x_i -i)
// Loss = f -0
// dL/dx_i = dL/df * df/dx_i =  exp (x_i-i)
void  LossConvexExample2::gradientCompute(){
    //symbol deduced formula to compute gradient to prevLayerPoint->m_pdYTensor
    Tensor<float> & prevY = *(m_prevLayer->m_pYTensor);
    Tensor<float> & prevdY = *(m_prevLayer->m_pdYTensor);
    long N = prevY.getLength();
    for (long i=0; i< N ;++i){
        prevdY[i] += exp ( prevY[i] - i);
    }
}

void  LossConvexExample2::printGroundTruth(){
    cout<<"For this specific Loss function: f(x) = \\sum exp(x_i -i), Ground Truth is: ";
    long N = m_prevLayer->m_pYTensor->getLength();
    cout<<"( ";
    for (long i=0; i< N; ++i){
        if (i != N-1 ) cout<<"-inf"<<", ";
        else cout<<"-inf";
    }
    cout<<" )"<<endl;
}
