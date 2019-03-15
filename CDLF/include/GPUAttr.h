//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_GPUATTR_H
#define CDLF_FRAMEWORK_GPUATTR_H

void cudaPrintError();

class GPUAttr{
public:
    GPUAttr();
    ~GPUAttr();

    static  int m_numSMs;
    static  int m_maxThreadsPerBlock;
    static  int m_blocksPerGrid;
    static  int m_numDevices;
    static  int m_driverVersion;


    void getGPUAttr();

    int m_computeCapabilityMajor;
    int m_computeCapabilityMinor;
};





#endif //CDLF_FRAMEWORK_GPUATTR_H
