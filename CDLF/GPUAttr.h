//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_GPUATTR_H
#define CDLF_FRAMEWORK_GPUATTR_H

static  int g_numSMs = 0;
static  int g_maxThhreasPerBlock = 0;
static  long g_blockCount = 0;


void getGPUAttr();

#endif //CDLF_FRAMEWORK_GPUATTR_H
