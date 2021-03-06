//
// Created by Hui Xie on 6/13/2018.
//

#include "LossConvexExample1.h"
#include <iostream>
#include <math.h>       /* pow */
#include <LossConvexExample1.h>

using namespace std;

LossConvexExample1::LossConvexExample1(const int id, const string& name,  Layer *prevLayer): LossLayer(id,name, prevLayer){
   m_type = "LossConvexExample1";
}
LossConvexExample1::~LossConvexExample1(){

}


float LossConvexExample1::lossCompute(){
    //use m_prevLayerPointer->m_pYTensor,
    m_loss = 0;
    Tensor<float> & prevY = *(m_prevLayer->m_pYTensor);
    int N = prevY.getLength();
    for (int i=0; i< N ;++i){
        m_loss += pow( prevY.e(i) - i , 2);
    }
    return m_loss;
}

// f= \sum (x_i-i)^2
// Loss = f-0
// dL/dx_i = dL/df * df/dx_i =  2* (x_i-i)
void LossConvexExample1::gradientCompute() {
    //symbol deduced formula to compute gradient to prevLayer->m_pdYTensor
    Tensor<float> &prevY = *(m_prevLayer->m_pYTensor);
    Tensor<float> &prevdY = *(m_prevLayer->m_pdYTensor);
    int N = prevY.getLength();
    for (int i = 0; i < N; ++i) {
        prevdY[i] += 2 * (prevY[i] - i);
    }
}

void  LossConvexExample1::printGroundTruth(){
    cout<<"For this specific Loss function, Ground Truth is: ";
    int N = m_prevLayer->m_pYTensor->getLength();
    cout<<"( ";
    for (int i=0; i< N; ++i){
        if (i != N-1 ) cout<<i<<", ";
        else cout<<i;
    }
    cout<<" )"<<endl;
}

