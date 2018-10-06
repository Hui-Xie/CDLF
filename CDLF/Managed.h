//
// Created by Hui Xie on 10/1/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_MANAGED_H
#define CDLF_FRAMEWORK_MANAGED_H

#include <cstdlib>
#ifdef Use_GPU
   #include <cuda_runtime.h>
#endif
#include "GPUAttr.h"

class Managed {
public:
    void *operator new(size_t len) {
        void *ptr = nullptr;
        #ifdef Use_GPU
            cudaMallocManaged(&ptr, len);
            cudaDeviceSynchronize();
        #else
            ptr = malloc(len);
        #endif
        return ptr;
    }

    void operator delete(void *ptr) {
        #ifdef Use_GPU
            cudaDeviceSynchronize();
            cudaFree(ptr);
        #else
            free(ptr);
        #endif
        ptr = nullptr;
    }
};


#endif //CDLF_FRAMEWORK_MANAGED_H
