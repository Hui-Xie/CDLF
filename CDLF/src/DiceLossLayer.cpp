//
// Created by Hui Xie on 02/22/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <DiceLossLayer.h>

DiceLossLayer::DiceLossLayer(const int id, const string &name, Layer *prevLayer) : LossLayer(id, name, prevLayer) {
    m_type = "DiceLossLayer";
    m_xDotgnorm = 0;
    m_sum_x_gnorm2 = 0;
}

DiceLossLayer::~DiceLossLayer() {
    //null;
}

float DiceLossLayer::lossCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    const int N = X.getLength();
    m_xDotgnorm = sqrt(X.hadamard(*m_pGroundTruth).normSquare());
    m_sum_x_gnorm2 = X.normSquare()+ m_pGroundTruth->normSquare();
    m_loss = 1 -2* m_xDotgnorm/ m_sum_x_gnorm2;
    return m_loss;
}

void DiceLossLayer::gradientCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    Tensor<float> & dX = *(m_prevLayer->m_pdYTensor);
    Tensor<float> & G = *m_pGroundTruth;
    Tensor<float> xDotg = X.hadamard(G);
    const int N = X.getLength();
    float sum_x_gnorm2_square = m_sum_x_gnorm2*m_sum_x_gnorm2;
    float a = 2.0*m_sum_x_gnorm2/m_xDotgnorm;
    for (int i=0; i<N; ++i){
        dX[i] += (4*m_xDotgnorm*X.e(i)-a*xDotg.e(i))/sum_x_gnorm2_square;
    }
}




