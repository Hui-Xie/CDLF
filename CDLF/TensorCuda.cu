//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "TensorCuda.h"
#include "DeviceKernels.h"
#include "GPUAttr.h"

void cudaZeroInitialize(float* m_data, const long N){
    deviceZeroInitialize<<<g_blockCount, g_maxThhreasPerBlock>>>(m_data, N);
    cudaDeviceSynchronize();
}
