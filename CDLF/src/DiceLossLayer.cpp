//
// Created by Hui Xie on 02/22/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <DiceLossLayer.h>

DiceLossLayer::DiceLossLayer(const int id, const string &name, Layer *prevLayer) : LossLayer(id, name, prevLayer) {
    m_type = "DiceLossLayer";

    if ("SigmoidLayer" != prevLayer->m_type){
        cout<<"Error: DiceLossLayer should follow with SigmoidLayer"<<endl;
        std:exit(EXIT_FAILURE);
    }
}

DiceLossLayer::~DiceLossLayer() {
    //null;
}

float DiceLossLayer::lossCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    const int N = X.getLength();

    // here nom is L1 norm, when x>0, g>0, L1norm = sum
    const float xDotg_norm = X.hadamard(*m_pGroundTruth).sum();
    const float xPlusg_norm = X.sum()+ m_pGroundTruth->sum();
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

    // here norm is L1 norm, when x>0, g>0, L1norm = sum
    const int N = X.getLength();
    const float xDotg_norm = X.hadamard(*m_pGroundTruth).sum();
    const float xPlusg_norm = X.sum()+ m_pGroundTruth->sum();
    const float xPlusg_norm2 = 2.0/(xPlusg_norm * xPlusg_norm);
    for (int i=0; i<N; ++i){
        dX[i] += (xDotg_norm - G.e(i)*xPlusg_norm)*xPlusg_norm2;
    }
}




