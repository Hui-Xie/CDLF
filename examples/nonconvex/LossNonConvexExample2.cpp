//
// Created by Hui Xie on 6/15/2018.
//

#include "LossNonConvexExample2.h"

#include <iostream>
using namespace std;

LossNonConvexExample2::LossNonConvexExample2(const int id, const string& name): LossLayer(id, name){
    //previousLayer's width must be 2 for this specific non-convex function
    cout<<"Notes: Make sure that final layer only 1 neuron."<<endl;
}
LossNonConvexExample2::~LossNonConvexExample2(){

}


float LossNonConvexExample2::lossCompute(Tensor<float>* pGroundTruth){
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    m_loss = X[0]*sin(X[0]);
}

//f(x) = x*sin(x)
//Loss = f -0;
//dL/dx = dL/df* df/dx = sin(x)+ x*cos(x)
void  LossNonConvexExample2::gradientCompute(){
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    Tensor<float> & dY = *(m_prevLayer->m_pdYTensor);
    dY[0] += sin(X[0])+X[0]*cos(X[0]);
}

void  LossNonConvexExample2::printGroundTruth(){
    //null
    return;
}