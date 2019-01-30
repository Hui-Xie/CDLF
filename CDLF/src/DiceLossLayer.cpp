//
// Created by Hui Xie on 02/22/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <DiceLossLayer.h>

DiceLossLayer::DiceLossLayer(const int id, const string &name, Layer *prevLayer) : LossLayer(id, name, prevLayer) {
    m_type = "DiceLossLayer";
}

DiceLossLayer::~DiceLossLayer() {
    //null;
}

float DiceLossLayer::lossCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    const int N = X.getLength();
    const float xDotg_norm = X.hadamard(*m_pGroundTruth).L2Norm();
    const float xPlusg_norm = X.L2Norm()+ m_pGroundTruth->L2Norm();
    if (0 == xPlusg_norm) {
        return 1;
    }
    m_loss = 1 -2.0* xDotg_norm/ xPlusg_norm;
    return m_loss;
}

void DiceLossLayer::gradientCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    Tensor<float> & dX = *(m_prevLayer->m_pdYTensor);
    Tensor<float> & G = *m_pGroundTruth;
    Tensor<float> xDotg = X.hadamard(G);
    const int N = X.getLength();

    const float xnorm = X.L2Norm();
    const float xDotg_norm = X.hadamard(*m_pGroundTruth).L2Norm();
    const float xPlusg_norm = X.L2Norm()+ m_pGroundTruth->L2Norm();

    if (0 == xnorm || 0 == xDotg_norm || 0 == xPlusg_norm){
        return ;
    }

    float a = xDotg_norm/xnorm;
    float b = xPlusg_norm/xDotg_norm;
    float xPlusg_norm2 = xPlusg_norm*xPlusg_norm;

    for (int i=0; i<N; ++i){
        dX[i] += 2.0*(a*X.e(i)-b*xDotg.e(i))/xPlusg_norm2;
    }
}




