//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_GPUATTR_H
#define CDLF_FRAMEWORK_GPUATTR_H



class GPUAttr{
public:
    GPUAttr();
    ~GPUAttr();

    static  int g_numSMs;
    static  int g_maxThreadsPerBlock;
    static  long g_blocksPerGrid;
    static  bool g_useGPU;

    void setUseGPU(const bool useGPU);

    void getGPUAttr();

};





#endif //CDLF_FRAMEWORK_GPUATTR_H
