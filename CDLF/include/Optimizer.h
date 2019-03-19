//
// Created by Hui Xie on 03/19/2019
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_OPTIMIZER_H
#define CDLF_FRAME_OPTIMIZER_H

#include "Tensor.h"

class AdamOptmizer {
public:
    AdamOptmizer(const float lr, const float beta1,  const float beta2);
    ~AdamOptmizer();

    void adam(int t, Tensor<float>* pM, Tensor<float>* pR, const Tensor<float>* pG,  Tensor<float>* pW);

    float m_lr;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
};


#endif // CDLF_FRAME_OPTIMIZER_H