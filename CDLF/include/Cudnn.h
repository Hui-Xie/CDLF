
//
// Created by Hui Xie on 01/24/2019.
// Copyright (c) 2019 Hui Xie. All rights reserved.


#ifndef CDLF_FRAME_CUDNN_H
#define CDLF_FRAME_CUDNN_H

#include "cudnn.h"
#include <cuda.h>
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

/*
#define checkCUDA(expression)                               \
  {                                                          \
    ​cudaError_t status = (expression);                     \
    if (status != cudaSuccess) {                    \
      std::cerr << "cuda Error: line " << __LINE__ << " of file: "<< __FILE__\
                << std::endl                                  \
                << cudaGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }
*/


struct GPUUsage{
    int m_deviceID;
    size_t m_freeMem; //in bytes;
    size_t m_totalMem; // in bytes;
};

class Cudnn{
public:
    Cudnn(Layer* pLayer);
    ~Cudnn();

    cudnnHandle_t m_cudnnContext;
    cudnnTensorDescriptor_t m_xDescriptor;
    cudnnTensorDescriptor_t m_yDescriptor;

    Layer* m_pLayer;

    int chooseIdleGPU();

    void allocateDeviceX();
    void allocateDeviceY(bool copyComputedY = false);
    void allocateDevicedX();
    void allocateDevicedY();

    void freeDeviceX();
    void freeDeviceY();
    void freeDevicedX();
    void freeDevicedY();

    virtual void setXDescriptor();
    virtual void setYDescriptor() = 0;
    virtual void setDescriptors() = 0;

    virtual void forward() = 0;
    virtual void backward(bool computeW, bool computeX) = 0;

    float* d_pX;
    float* d_pY;
    float* d_pdX;
    float* d_pdY;
    
protected:
    void checkDeviceTensor(float* d_pA, const vector<int>  tensorSize);


};


#endif //CDLF_FRAME_CUDNN_H
