//
// Created by Hui Xie on 10/1/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_MANAGED_H
#define CDLF_FRAMEWORK_MANAGED_H

#include <cstdlib>
#include <cuda_runtime.h>
#include "GPUAttr.h"

class Managed {
public:
    void *operator new(size_t len) {
        void *ptr = nullptr;
        if (GPUAttr::g_useGPU){
            cudaMallocManaged(&ptr, len);
            cudaDeviceSynchronize();
        }
        else{
            ptr = malloc(len);
        }

        return ptr;
    }

    void operator delete(void *ptr) {
        if (GPUAttr::g_useGPU){
            cudaDeviceSynchronize();
            cudaFree(ptr);
        }
        else{
            free(ptr);
        }
        ptr = nullptr;
    }
};


#endif //CDLF_FRAMEWORK_MANAGED_H
