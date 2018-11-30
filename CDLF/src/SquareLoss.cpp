//
// Created by Hui Xie on 11/29/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <SquareLoss.h>

#include "SquareLoss.h"

SquareLoss::SquareLoss(const int id, const string &name, Layer *prevLayer) : LossLayer(id, name, prevLayer) {
    m_type = "SquareLoss";
}

SquareLoss::~SquareLoss() {
   //null;
}

void SquareLoss::printGroundTruth() {
   m_pGroundTruth->print();
}

// L= 0.5*\sum (x_i- g_i)^2
float SquareLoss::lossCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    const long N = X.getLength();
    float loss = 0;
    for (long i=0; i<N; ++i){
        loss += pow(X.e(i) - m_pGroundTruth->e(i), 2);
    }
    m_loss = loss*0.5;
    return m_loss;
 }

//dL/dx = (x_i-g_i)
void SquareLoss::gradientCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    Tensor<float> & dX = *(m_prevLayer->m_pdYTensor);
    const long N = X.getLength();
    for (long i=0; i<N; ++i){
       dX[i] = X.e(i) - m_pGroundTruth->e(i);
    }
}
