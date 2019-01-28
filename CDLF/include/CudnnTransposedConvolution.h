//
// Created by Hui Xie on 01/28/2019.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_CUDNNTRANSPOSEDCONVOLUTION_H
#define RL_NONCONVEX_CUDNNTRANSPOSEDCONVOLUTION_H

#include <CudnnBasicConvolution.h>
#include <TransposedConvolutionLayer.h>


class CudnnTransposedConvolution : public CudnnBasicConvolution{
public:
    CudnnTransposedConvolution(TransposedConvolutionLayer* pLayer, const vector<int>& filterSize, const int numFilters=1, const int stride =1);
    ~CudnnTransposedConvolution();

    virtual void setForwardAlg();
    virtual void setBackwardDataAlg();
    virtual void setBackWardFilterAlg();

    virtual void setYDescriptor();
    virtual void forward();
    virtual void backward(bool computeW, bool computeX);
};

#endif //RL_NONCONVEX_CUDNNTRANSPOSEDCONVOLUTION_H