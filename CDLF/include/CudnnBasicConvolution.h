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
    cudnnTensorDescriptor_t m_bDescriptor;
    cudnnConvolutionFwdAlgo_t m_fwdAlg;
    cudnnConvolutionBwdDataAlgo_t m_bwdDataAlg;
    cudnnConvolutionBwdFilterAlgo_t m_bwdFilterAlg;

    virtual void setXDescriptor();
    void setConvDescriptor();
    void setBDescriptor();

    virtual void setDescriptors();
    void allocateDeviceW();
    void allocateDevicedW();
    void allocateDeviceB();
    void allocateDevicedB();

    void freeDeviceW();
    void freeDevicedW();
    void freeDeviceB();
    void freeDevicedB();
    void freeWorkSpace();

    virtual void setForwardAlg()=0;
    virtual void setBackwardDataAlg()=0;
    virtual void setBackWardFilterAlg()=0;

    virtual void setWDescriptor() =0;

    size_t m_workspaceSize;
    void* d_pWorkspace;

    float* d_pW;
    float* d_pdW;
    float* d_pB; //bias
    float* d_pdB;

};

#endif //RL_NONCONVEX_CUDNNCONVOLUTIONBASIC_H