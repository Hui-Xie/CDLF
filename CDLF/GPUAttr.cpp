//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "GPUAttr.h"
#include "cuda_runtime.h"
#include <iostream>
using namespace std;

void getGPUAttr(){
    cudaDeviceGetAttribute ( &g_numSMs, cudaDevAttrMultiProcessorCount, 0);
    cout<<"g_numSMs = "<< g_numSMs<<endl;

    cudaDeviceGetAttribute ( &g_maxThhreasPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    cout<<"g_maxThhreasPerBlock = "<< g_maxThhreasPerBlock<<endl;

    g_blockCount = 32* g_numSMs;



}