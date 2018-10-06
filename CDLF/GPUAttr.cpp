//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "GPUAttr.h"
#ifdef UseGPU
   #include "cuda_runtime.h"
#endif
#include <iostream>

using namespace std;

int GPUAttr::m_numSMs = 0;
int GPUAttr::m_maxThreadsPerBlock = 0;
long GPUAttr::m_blocksPerGrid = 0;

GPUAttr::GPUAttr() {

}

GPUAttr::~GPUAttr() {

}

void GPUAttr::getGPUAttr() {
#ifdef UseGPU
    cudaDeviceGetAttribute(&m_numSMs, cudaDevAttrMultiProcessorCount, 0);
    cout << "m_numSMs = " << m_numSMs << endl;

    cudaDeviceGetAttribute(&m_maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    cout << "m_maxThreadsPerBlock = " << m_maxThreadsPerBlock << endl;

    m_blocksPerGrid = 32 * m_numSMs;
#endif

}