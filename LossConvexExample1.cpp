//
// Created by Sheen156 on 6/13/2018.
//

#include "LossConvexExample1.h"
#include <iostream>
using namespace std;

LossConvexExample1::LossConvexExample1(Layer* preLayer): LossLayer(preLayer){

}
LossConvexExample1::~LossConvexExample1(){

}


float LossConvexExample1::lossCompute(){
    //use m_prevLayerPointer->m_pYVector,
    m_loss = 0;
    long N = m_prevLayerPointer->m_pYVector->size();
    DynamicVector<float> & prevY = *(m_prevLayerPointer->m_pYVector);
    for (long i=0; i< N ;++i){
        m_loss += pow( prevY[i] - i , 2);
    }
    return m_loss;
}

// f= \sum (x_i-i)^2
// Loss = f-0
// dL/dx_i = dL/df * df/dx_i =  2* (x_i-i)
void  LossConvexExample1::gradientCompute(){
    //symbol deduced formula to compute gradient to prevLayerPoint->m_pdYVector
    long N = m_prevLayerPointer->m_pYVector->size();
    DynamicVector<float> & prevY = *(m_prevLayerPointer->m_pYVector);
    DynamicVector<float> & prevdY = *(m_prevLayerPointer->m_pdYVector);
    for (long i=0; i< N ;++i){
        prevdY[i] = 2 * ( prevY[i] - i);
    }
}

void  LossConvexExample1::printGroundTruth(){
    cout<<"For this specific Loss function, Ground Truth is: ";
    long N = m_prevLayerPointer->m_width;
    cout<<"( ";
    for (long i=0; i< N; ++i){
        if (i != N-1 ) cout<<i<<", ";
        else cout<<i;
    }
    cout<<" )"<<endl;
}