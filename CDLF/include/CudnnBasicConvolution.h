//
// Created by Hui Xie on 01/28/2019.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_CUDNNCONVOLUTIONBASIC_H
#define RL_NONCONVEX_CUDNNCONVOLUTIONBASIC_H

#include "Cudnn.h"
#include <ConvolutionBasicLayer.h>


class CudnnBasicConvolution : public Cudnn{
public:
    CudnnBasicConvolution(ConvolutionBasicLayer* pLayer, const vector<int>& filterSize, const int numFilters=1, const int stride =1);
    ~CudnnBasicConvolution();

    cudnnConvolutionDescriptor_t    m_convDescriptor;
    cudnnFilterDescriptor_t m_wDescriptor;
    cudnnConvolutionFwdAlgo_t m_fwdAlg;
    cudnnConvolutionBwdDataAlgo_t m_bwdDataAlg;
    cudnnConvolutionBwdFilterAlgo_t m_bwdFilterAlg;


    vector<int> m_filterSize;
    int m_numFilters;
    int m_stride;

    size_t m_workspaceSize;

    void setXDescriptor();
    void setConvDescriptor();

    void setDescriptors();
    void allocateDeviceX();
    void allocateDeviceY();
    void allocateDeviceW();
    void allocateDevicedX();
    void allocateDevicedY();
    void allocateDevicedW();

    virtual void setForwardAlg()=0;
    virtual void setBackwardDataAlg()=0;
    virtual void setBackWardFilterAlg()=0;

    virtual void setWDescriptor() =0;
    virtual void setYDescriptor() = 0;

    virtual void forward() = 0;
    virtual void backward(bool computeW, bool computeX)=0;

    void* d_pWorkspace;
    float* d_pX;
    float* d_pY;
    float* d_pW;
    float* d_pdX;
    float* d_pdY;
    float* d_pdW;
};

#endif //RL_NONCONVEX_CUDNNCONVOLUTIONBASIC_H