//
// Created by Hui Xie on 01/28/2019.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_CUDNNTRANSPOSEDCONVOLUTION_H
#define CDLF_FRAME_CUDNNTRANSPOSEDCONVOLUTION_H

#include <CudnnBasicConvolution.h>
#include <TransposedConvolutionLayer.h>


class CudnnTransposedConvolution : public CudnnBasicConvolution{
public:
    CudnnTransposedConvolution(TransposedConvolutionLayer* pLayer);
    ~CudnnTransposedConvolution();

    virtual void setForwardAlg();
    virtual void setBackwardDataAlg();
    virtual void setBackWardFilterAlg();

    virtual void setWDescriptor();
    virtual void setYDescriptor();

    virtual void forward();
    virtual void backward(bool computeW, bool computeX);
};

#endif //CDLF_FRAME_CUDNNTRANSPOSEDCONVOLUTION_H