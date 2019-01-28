
//
// Created by Hui Xie on 01/24/2019.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_CUDNNCONVOLUTION_H
#define RL_NONCONVEX_CUDNNCONVOLUTION_H

#include "CudnnBasicConvolution.h"
#include "ConvolutionLayer.h"


class CudnnConvolution : public CudnnBasicConvolution{
public:
    CudnnConvolution(ConvolutionLayer* pLayer, const vector<int>& filterSize, const int numFilters=1, const int stride =1);
    ~CudnnConvolution();

    virtual void setForwardAlg();
    virtual void setBackwardDataAlg();
    virtual void setBackWardFilterAlg();

    virtual void setWDescriptor();
    virtual void setYDescriptor();

    virtual void forward();
    virtual void backward(bool computeW, bool computeX);
};

#endif //RL_NONCONVEX_CUDNNCONVOLUTION_H