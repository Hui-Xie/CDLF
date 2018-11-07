//
// Created by Hui Xie on 6/13/2018.
//

#include "LossNonConvexExample1.h"
#include <iostream>
using namespace std;

LossNonConvexExample1::LossNonConvexExample1(const int id, const string& name,  Layer *prevLayer): LossLayer(id,name, prevLayer){
    m_type = "LossNonConvexExample1";
    //previousLayer's width must be 2 for this specific non-convex function
    cout<<"Notes: Make sure that final layer only 2 neurons."<<endl;
}
LossNonConvexExample1::~LossNonConvexExample1(){

}


float LossNonConvexExample1::lossCompute(){
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    m_loss = 3*X[1]*sin(X[0])+5*X[0]*cos(X[1])+0.5*X[0]*X[1]+X[0]*X[0]-X[1]*X[1];
    return m_loss;
}

//f(x,y) = 3ysin(x)+5xcos(y)+0.5xy+x^2-y^2
//Loss = f -0;
//dL/dx = dL/df* df/dx = 3ycos(x)+ 5cos(y) + 0.5y +2x
//dL/dy = dL/df* df/dy = 3sin(x)- 5xsin(y)+ 0.5x-2y
void  LossNonConvexExample1::gradientCompute(){
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    Tensor<float> & dY = *(m_prevLayer->m_pdYTensor);
    dY[0] += 3*X[1]*cos(X[0])+5*cos(X[1])+0.5*X[1]+2*X[0];
    dY[1] += 3*sin(X[0])-5*X[0]*sin(X[1])+0.5*X[0]-2*X[1];
}

void  LossNonConvexExample1::printGroundTruth(){
    //null
    return;
}