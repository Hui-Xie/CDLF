//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_TENSORDEVICE_H
#define CDLF_FRAMEWORK_TENSORDEVICE_H

//cuda is a bridge between Tensor and Device



void cudaInitialize(float* m_data, const long N, const float value=0);

// C = A*B, where A has a size of M*K, B has a size of K*N, C will has a size of M*N
void cuda2DMatrixProduct(float* pA, float* pB, float* pC, const long M,const long N, const long K);

// B = A', where B has a size M*N
void cuda2DMatrixTranspose(float* pA, float* pB, const long M, const long N);


#endif //CDLF_FRAMEWORK_TENSORDEVICE_H
