
//
// Created by Hui Xie on 01/24/2019.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_CUDNNCONVOLUTION_H
#define CDLF_FRAME_CUDNNCONVOLUTION_H

#include "CudnnBasicConvolution.h"
#include "ConvolutionLayer.h"


class CudnnConvolution : public CudnnBasicConvolution{
public:
    CudnnConvolution(ConvolutionLayer* pLayer);
    ~CudnnConvolution();

    virtual void setForwardAlg();
    virtual void setBackwardDataAlg();
    virtual void setBackWardFilterAlg();

    virtual void setWDescriptor();
    virtual void setYDescriptor();

    virtual void forward();
    virtual void backward(bool computeW, bool computeX);
};

#endif //CDLF_FRAME_CUDNNCONVOLUTION_H