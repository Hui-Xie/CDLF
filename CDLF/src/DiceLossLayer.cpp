//
// Created by Hui Xie on 02/22/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <DiceLossLayer.h>

DiceLossLayer::DiceLossLayer(const int id, const string &name, Layer *prevLayer) : LossLayer(id, name, prevLayer) {
    m_type = "DiceLossLayer";
    m_xDotg_norm = 0;
    m_xPlusg_norm = 0;
}

DiceLossLayer::~DiceLossLayer() {
    //null;
}

float DiceLossLayer::lossCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    const int N = X.getLength();
    m_xDotg_norm = X.hadamard(*m_pGroundTruth).L2Norm();
    m_xPlusg_norm = X.L2Norm()+ m_pGroundTruth->L2Norm();
    m_loss = 1 -2.0* m_xDotg_norm/ m_xPlusg_norm;
    return m_loss;
}

void DiceLossLayer::gradientCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    Tensor<float> & dX = *(m_prevLayer->m_pdYTensor);
    Tensor<float> & G = *m_pGroundTruth;
    Tensor<float> xDotg = X.hadamard(G);
    const int N = X.getLength();

    float xnorm = X.L2Norm();
    float a = m_xDotg_norm/xnorm;
    float b = m_xPlusg_norm/m_xDotg_norm;
    float xPlusg_norm2 = m_xPlusg_norm*m_xPlusg_norm;

    for (int i=0; i<N; ++i){
        dX[i] += 2.0*(a*X.e(i)-b*xDotg.e(i))/xPlusg_norm2;
    }
}




