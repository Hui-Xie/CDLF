//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "TensorCuda.h"
#include "DeviceKernels.h"
#include "GPUAttr.h"

void cudaInitialize(float* m_data, const long N, const float value){
    deviceInitialize<<<g_blockCount, g_maxThhreasPerBlock>>>(m_data, N, value);
    cudaDeviceSynchronize();
}
