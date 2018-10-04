//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_GPUATTR_H
#define CDLF_FRAMEWORK_GPUATTR_H

static  int g_numSMs = 0;
static  int g_maxThreadsPerBlock = 0;
static  long g_blocksPerGrid = 0;
static  bool g_useGPU = true;

void setUseGPU(const bool useGPU);

void getGPUAttr();

#endif //CDLF_FRAMEWORK_GPUATTR_H
