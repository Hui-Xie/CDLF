//
// Created by Hui Xie on 01/28/2019.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_CUDNNCONVOLUTIONBASIC_H
#define RL_NONCONVEX_CUDNNCONVOLUTIONBASIC_H

#include "Cudnn.h"
#include <ConvolutionBasicLayer.h>


class CudnnBasicConvolution : public Cudnn{
public:
    CudnnBasicConvolution(ConvolutionBasicLayer* pLayer);
    ~CudnnBasicConvolution();

    cudnnConvolutionDescriptor_t    m_convDescriptor;
    cudnnFilterDescriptor_t m_wDescriptor;
    cudnnConvolutionFwdAlgo_t m_fwdAlg;
    cudnnConvolutionBwdDataAlgo_t m_bwdDataAlg;
    cudnnConvolutionBwdFilterAlgo_t m_bwdFilterAlg;

    virtual void setXDescriptor();
    void setConvDescriptor();

    virtual void setDescriptors();
    void allocateDeviceW();
    void allocateDevicedW();

    void freeDeviceW();
    void freeDevicedW();
    void freeWorkSpace();

    virtual void setForwardAlg()=0;
    virtual void setBackwardDataAlg()=0;
    virtual void setBackWardFilterAlg()=0;

    virtual void setWDescriptor() =0;

    size_t m_workspaceSize;
    void* d_pWorkspace;

    float* d_pW;
    float* d_pdW;

};

#endif //RL_NONCONVEX_CUDNNCONVOLUTIONBASIC_H