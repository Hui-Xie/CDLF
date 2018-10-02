//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "TensorCuda.h"
#include "DeviceKernels.h"

void cudaZeroInitialize(float* m_data, const long N){
    deviceZeroInitialize<<<N+1023/1024, 1024>>>(m_data, N);
    
}
