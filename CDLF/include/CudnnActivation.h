//
// Created by Hui Xie on 01/29/2019.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_CUDNNACTIVATION_H
#define RL_NONCONVEX_CUDNNACTIVATION_H

#include "Cudnn.h"

/*
 * activation mode in cudnn.h

 typedef enum
{
    CUDNN_ACTIVATION_SIGMOID      = 0,
    CUDNN_ACTIVATION_RELU         = 1,
    CUDNN_ACTIVATION_TANH         = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3,
    CUDNN_ACTIVATION_ELU          = 4
} cudnnActivationMode_t;

*/

class CudnnActivation : public Cudnn{
public:
    CudnnActivation(Layer* pLayer, cudnnActivationMode_t activationMode);
    ~CudnnActivation();

    cudnnActivationDescriptor_t    m_activationDescriptor;
    cudnnActivationMode_t          m_activationMode;

    void setActivationDescriptor();

    virtual void setYDescriptor();
    virtual void setDescriptors();

    virtual void forward();
    virtual void backward(bool computeW, bool computeX);

};

#endif //RL_NONCONVEX_CUDNNACTIVATION_H