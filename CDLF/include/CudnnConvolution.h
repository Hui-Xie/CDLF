
//
// Created by Hui Xie on 01/24/2019.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_CUDNNCONVOLUTION_H
#define RL_NONCONVEX_CUDNNCONVOLUTION_H

#include "Cudnn.h"
#include "ConvolutionLayer.h"


class CudnnConvolution : public Cudnn{
public:
    CudnnConvolution(ConvolutionLayer* pLayer, const vector<int>& filterSize, const int numFilters=1, const int stride =1);
    ~CudnnConvolution();

    cudnnConvolutionDescriptor_t    m_convDescriptor;
    cudnnFilterDescriptor_t m_filterDescriptor;
    cudnnConvolutionFwdAlgo_t m_fwdConvAlgorithm;

    vector<int> m_filterSize;
    int m_numFilters;
    int m_stride;

    size_t m_workspaceSize;

    void setXDescriptor();
    void setFilterDescriptor();
    void setConvDescriptor();
    void setYDescriptor();
    void setForwardAlg();
    void allocateDeviceMem();

    void setDescriptorsAndAlg();


    void forward();
    void backward();

    void* d_pWorkspace;
    float* d_pX;
    float* d_pY;
    float* d_pFilter;


};

#endif //RL_NONCONVEX_CUDNNCONVOLUTION_H