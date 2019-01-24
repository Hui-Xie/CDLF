
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
    int m_numFilters;
    vector<int> m_filterSize;
    int* m_filterSizeArray; // include the number of Filter dimension
    int m_filterArrayDim;
    size_t m_workspaceSize;

    void setConvDescriptorsAndAlgorithm();
    void allocateDeviceMemAndCopy();
    bool isOutputDimCorrect();

    void forward();
    void backward();

    void* d_pWorkspace;
    float* d_pX;
    float* d_pY;
    float* d_pFilter;


};

#endif //RL_NONCONVEX_CUDNNCONVOLUTION_H