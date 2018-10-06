//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "GPUAttr.h"
#include "cuda_runtime.h"
#include <iostream>
using namespace std;

GPUAttr::GPUAttr(){
    g_numSMs =0;
    g_maxThreadsPerBlock = 0;
    g_blocksPerGrid = 0;
    g_useGPU = true;
}

GPUAttr::~GPUAttr(){

}

void GPUAttr::setUseGPU(const bool useGPU){
    g_useGPU = useGPU;
    if (GPUAttr::g_useGPU){
        cout<<"Info: program will use GPU."<<endl;
    }
    else{
        cout<<"Info: program will use CPU, instead of GPU." <<endl;
    }
}

void GPUAttr::getGPUAttr(){
    if (GPUAttr::g_useGPU){
        cudaDeviceGetAttribute ( &g_numSMs, cudaDevAttrMultiProcessorCount, 0);
        cout<<"g_numSMs = "<< g_numSMs<<endl;

        cudaDeviceGetAttribute ( &g_maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
        cout<<"g_maxThreadsPerBlock = "<< g_maxThreadsPerBlock<<endl;

        g_blocksPerGrid = 32* g_numSMs;
    }

}