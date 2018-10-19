//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "GPUAttr.h"
#ifdef Use_GPU
   #include "cuda_runtime.h"
#endif
#include <iostream>

using namespace std;

int GPUAttr::m_numSMs = 0;
int GPUAttr::m_maxThreadsPerBlock = 0;
long GPUAttr::m_blocksPerGrid = 0;

void cudaPrintError(){
#ifdef Use_GPU
    cudaError_t cudaError = cudaGetLastError();
    if (0 != cudaError){
        cout<<"Cuda error: "<<cudaError<< "; Error string = "<< cudaGetErrorString(cudaError)<<endl;
    }
#endif
}

GPUAttr::GPUAttr() {

}

GPUAttr::~GPUAttr() {

}

void GPUAttr::getGPUAttr() {
#ifdef Use_GPU
    cudaDeviceReset();

    cudaDeviceGetAttribute(&m_numSMs, cudaDevAttrMultiProcessorCount, 0);
    cout << "m_numSMs = " << m_numSMs << endl;

    cudaDeviceGetAttribute(&m_maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    cout << "m_maxThreadsPerBlock = " << m_maxThreadsPerBlock << endl;

    m_blocksPerGrid = 32 * m_numSMs;

    int deviceID = 0;
    if (cudaSuccess == cudaGetDevice(&deviceID)){
        cout<<"Currently use GPU device ID: "<<deviceID<<endl;
    }
    else{
        cudaPrintError();
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);
    int m_computeCapabilityMajor = deviceProp.major;
    int m_computeCapabilityMinor = deviceProp.minor;
    std::printf("GPU Compute Capability: %d.%d\n", m_computeCapabilityMajor, m_computeCapabilityMinor);

#endif

}