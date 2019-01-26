
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
      std::cerr << "cudnn Error: line " << __LINE__ << " of file: "<< __FILE__\
                << std::endl                                  \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }



class Cudnn{
public:
    Cudnn(Layer* pLayer);
    ~Cudnn();

    cudnnHandle_t m_cudnnContext;
    cudnnTensorDescriptor_t m_xDescriptor;
    cudnnTensorDescriptor_t m_yDescriptor;

    Layer* m_pLayer;

};

#endif //RL_NONCONVEX_CUDNN_H
