
//
// Created by Hui Xie on 01/24/2019.
// Copyright (c) 2018 Hui Xie. All rights reserved.


#ifndef RL_NONCONVEX_CUDNN_H
#define RL_NONCONVEX_CUDNN_H

#include "cudnn.h"
#include "Layer.h"

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << " of File: "<< __FILE__\
                << std::endl                                  \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }



class Cudnn{
public:
    Cudnn(Layer* pLayer, const int stride =1);
    ~Cudnn();

    cudnnHandle_t m_cudnnContext;
    cudnnTensorDescriptor_t m_xDescriptor;
    cudnnTensorDescriptor_t m_yDescriptor;

    Layer* m_pLayer;
    int m_stride;

    void setDescriptors();

protected:
    void getDimsArrayFromTensorSize(const vector<int> tensorSize, int*& array, int& dim);
    void getDimsArrayFromFilterSize(const vector<int> filterSize, const int numFilters, int*& array, int& dim);
    void generateStrideArray(const int stride, const int dim, int*& array);
};

#endif //RL_NONCONVEX_CUDNN_H
