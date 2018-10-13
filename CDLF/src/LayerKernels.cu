//
// Created by Hui Xie on 10/10/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "LayerKernels.h"

__global__ void deviceSigmoidDerivative(float* pX, float* pdY, const int k, float* pdX, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x; //i: thread index
    while (i < N){
        float  expx = exp(pX[i]);
        pdX[i] += pdY[i]*k*expx/pow(1+expx,2);
        i += blockDim.x*gridDim.x;
    }
}

__global__ void deviceSigmoid(float* pX, float* pY, const int k, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x; //i: thread index
    while (i < N){
        float exp_x = exp(-pX[i]);
        pY[i] = k/(1+exp_x);
        i += blockDim.x*gridDim.x;
    }

}


__global__ void deviceCrossEntropyGradient(float* pX, float* pGTX, float* pdX, const float epsilon, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x; //i: thread index
    while (i < N){
        if (0 != pX[i]){
            pdX[i] -= pGTX[i]/pX[i];
        }
        else{
            pdX[i] -= pGTX[i]/epsilon;
        }
        i += blockDim.x*gridDim.x;
    }
}


//C = A where A and C has different value type
__global__ void deviceElementCopy(unsigned char* pA,float* pC, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = (float)pA[i];
        i += blockDim.x*gridDim.x;
    }
}

//C = A if A>=0; C =0 else
__global__ void deviceRelu(float* pA,float* pC, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        if (pA[i]>0){
            pC[i] = pA[i];
        }
        else{
            pC[i] = 0;
        }
        i += blockDim.x*gridDim.x;
    }
}

// dL/dx = dL/dy * dy/dx = dL/dy if X>=0; 0 if X < 0
__global__ void deviceReluDerivative(float* pX,float* pdY, float* pdX, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        if (pX[i]>= 0){
            pdX[i] = pdY[i];
        }
        else{
            pdX[i] = 0;
        }
        i += blockDim.x*gridDim.x;
    }

}

__global__ void deviceSoftmax(float* pX, float* pY, const int nSoftmax, const long N){
    long j = threadIdx.x + blockIdx.x * blockDim.x; // j is thread index
    while (j < N){
        float sumExpX = 1e-8;;
        for (int i=0; i< nSoftmax; ++i){
            sumExpX += exp(pX[i*N+j]);
        }
        for (int i=0; i< nSoftmax; ++i){
            pY[i*N+j] = exp(pX[i*N+j])/sumExpX;
        }
        j += blockDim.x*gridDim.x;
    }
}

__global__ void deviceSoftmaxDerivative(float* pX,float* pdY, float* pdX, const int nSoftmax, const long N){
    long j = threadIdx.x + blockIdx.x * blockDim.x; // j is thread index
    while (j < N){
        float sumExpX = 1e-8;
        for (int i=0; i< nSoftmax; ++i){
            sumExpX += exp(pX[i*N+j]);
        }
        float sumExpX2 = sumExpX*sumExpX;

        // \sum(dL/dy_j*exp(x_j)
        float dyDotExpX = 0;
        for(int i=0; i< nSoftmax; ++i){
            dyDotExpX += pdY[i*N+j]*exp(pX[i*N+j]);
        }

        for(int i=0; i< nSoftmax; ++i){
            pdX[i*N+j] += exp(pX[i*N+j])*(pdY[i*N+j]*sumExpX-dyDotExpX)/sumExpX2;
        }

        j += blockDim.x*gridDim.x;
    }
}
