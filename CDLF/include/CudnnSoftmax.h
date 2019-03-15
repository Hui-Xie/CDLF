//
// Created by Hui Xie on 02/04/2019.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_CUDNNSOFTMAX_H
#define CDLF_FRAME_CUDNNSOFTMAX_H

#include "Cudnn.h"

/*
typedef enum
{
    CUDNN_SOFTMAX_FAST     = 0,              // straightforward implementation
    CUDNN_SOFTMAX_ACCURATE = 1,             // subtract max from every point to avoid overflow
    CUDNN_SOFTMAX_LOG      = 2
} cudnnSoftmaxAlgorithm_t;

typedef enum
{
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,   // compute the softmax over all C, H, W for each N
    CUDNN_SOFTMAX_MODE_CHANNEL = 1     // compute the softmax over all C for each H, W, N
} cudnnSoftmaxMode_t;

*/

class CudnnSoftmax : public Cudnn{
public:
    CudnnSoftmax(Layer* pLayer);
    ~CudnnSoftmax();

    cudnnSoftmaxAlgorithm_t          m_algorithm;
    cudnnSoftmaxMode_t               m_mode;

    virtual void setXDescriptor();
    virtual void setYDescriptor();
    virtual void setDescriptors();

    virtual void forward();
    virtual void backward(bool computeW, bool computeX);

};

#endif //CDLF_FRAME_CUDNNSOFTMAX_H