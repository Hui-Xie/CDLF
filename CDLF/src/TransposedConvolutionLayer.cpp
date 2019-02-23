//
// Created by Hui Xie on 11/19/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "TransposedConvolutionLayer.h"
#include <thread>
#include <TransposedConvolutionLayer.h>

#ifdef Use_GPU
   #include <CudnnTransposedConvolution.h>
#endif

TransposedConvolutionLayer::TransposedConvolutionLayer(const int id, const string &name, Layer *prevLayer,
                                                       const vector<int> &filterSize,
                                                       const vector<int>& stride, const int numFilters)
        : ConvolutionBasicLayer(id, name, prevLayer, filterSize, stride, numFilters) {
    m_type = "TransposedConvolutionLayer";
    updateTensorSize();
    constructFiltersAndY();
}

TransposedConvolutionLayer::~TransposedConvolutionLayer() {
    //the basic class is responsible for deleting memory
    //null
}


// For transposed convolution layer: outputTensorSize = (InputTensorSize -1)*stride + filterSize;
void TransposedConvolutionLayer::updateTensorSize() {
    m_tensorSize = m_prevLayer->m_tensorSize;
    const int dim = m_tensorSize.size();

    const int s = (1 == m_numInputFeatures)? 0: 1; //indicate whether previousTensor includes feature dimension

    for (int i = 0; i+s < dim; ++i) {
        m_tensorSize[i+s] = (m_tensorSize[i+s] - 1) * m_stride[i] + m_filterSize[i];
    }

    if (1 ==s){
        m_tensorSize[0] = 1;
    }

    m_tensorSizeBeforeCollapse = m_tensorSize;
    if (1 != m_numFilters) {
        m_tensorSize.insert(m_tensorSize.begin(), m_numFilters);
    }
    deleteOnes(m_tensorSize);
}


// Y = W*X+b
void TransposedConvolutionLayer::forward() {

#ifdef Use_GPU
    CudnnTransposedConvolution cudnnTConvolution(this);
    cudnnTConvolution.forward();

#else
    const int N = length(m_tensorSize) / m_numFilters;
    const vector<int> dimsSpanBeforeCollpase = genDimsSpan(m_tensorSizeBeforeCollapse);
    Tensor<float> *pExtendX = nullptr;

    const vector<int> filterSize = (1 == m_numInputFeatures)? m_filterSize : m_feature_filterSize;
    const vector<int> stride = (1 == m_numInputFeatures)? m_stride : m_feature_stride;

    m_prevLayer->m_pYTensor->dilute(pExtendX, m_prevLayer->m_tensorSize, filterSize, stride);

    const int nThread = (CPUAttr::m_numCPUCore+ m_numFilters-1)/m_numFilters;  //num of thread for each filter
    const int NRange = (N +nThread -1)/nThread;

    vector<std::thread> threadVec;
    for (int idxF = 0; idxF < m_numFilters; ++idxF) {
        for (int t= 0; t< nThread; ++t){  // th indicates thread
            threadVec.push_back(thread(
                    [this, idxF, t, N, &dimsSpanBeforeCollpase, pExtendX, NRange, filterSize]() {
                        Tensor<float> subX(filterSize);
                        const int offseti = idxF*N;
                        const vector<int> stride1(filterSize.size(),1);
                        const float bias = m_pB->e(idxF);
                        for (int i = NRange*t; i<NRange*(t+1) && i < N; ++i) {
                            pExtendX->subTensorFromTopLeft(m_pYTensor->offset2Index(dimsSpanBeforeCollpase, i), &subX, stride1);
                            m_pYTensor->e(offseti+i) = subX.conv(*m_pW[idxF]) + bias;
                        }

                    }
            ));

        }
    }

    for (int i = 0; i < threadVec.size(); ++i) {
        threadVec[i].join();
    }

    if (nullptr != pExtendX) {
        delete pExtendX;
        pExtendX = nullptr;
    }

#endif

}

// Y =W*X
// dL/dW = dL/dY * dY/dW;
// dL/dX = dL/dY * dY/dX;
// algorithm ref: https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
void TransposedConvolutionLayer::backward(bool computeW, bool computeX) {

#ifdef Use_GPU

    CudnnTransposedConvolution cudnnTConvolution(this);
    cudnnTConvolution.backward(computeW, computeX);

#else

    // dX needs to consider the accumulation of different filters
    if (1 != m_numFilters) {
        // ==================multithread computation=======================
        // allocate pdX and pdY along the filters
        Tensor<float> **pdX = (Tensor<float> **) new void *[m_numFilters];
        Tensor<float> **pdY = (Tensor<float> **) new void *[m_numFilters];
        Tensor<float> **pExpandDY = (Tensor<float> **) new void *[m_numFilters];
        for (int i = 0; i < m_numFilters; ++i) {
            pdX[i] = new Tensor<float>(m_prevLayer->m_tensorSize);
            pdX[i]->zeroInitialize();  // this is a necessary step as computeX use += operator
            pdY[i] = nullptr;//pdY memory will be allocated in the extractLowerDTensor function
            pExpandDY[i] = nullptr;//pExpandDY memory will be allocated in the dilute method;
        }
        vector<std::thread> threadVec;
        for (int idxF = 0; idxF < m_numFilters; ++idxF) {
            threadVec.push_back(thread(
                    [this, idxF, pdY, pExpandDY, computeW, computeX, pdX]() {
                        this->m_pdYTensor->extractLowerDTensor(idxF, pdY[idxF]);
                        if (computeW) {
                            this->computeDW(pdY[idxF], this->m_pdW[idxF]);
                            this->computeDb(pdY[idxF], idxF);
                        }
                        const vector<int> stride1(m_filterSize.size(),1);
                        if (computeX){
                            pdY[idxF]->dilute(pExpandDY[idxF], m_tensorSizeBeforeCollapse, m_feature_filterSize, m_feature_stride);
                            this->computeDX(pExpandDY[idxF], this->m_pW[idxF],pdX[idxF]); //as pdX needs to accumulate, pass pointer
                        }
                    }
            ));
        }

        for (int t = 0; t < threadVec.size(); ++t) {
            threadVec[t].join();
        }

        // accumulate pdX
        for (int idxF = 0; idxF < m_numFilters; ++idxF) {
            *m_prevLayer->m_pdYTensor += *pdX[idxF];
        }

        // free pdY and pdX
        for (int i = 0; i < m_numFilters; ++i) {
            if (nullptr != pdY[i]) {
                delete pdY[i];
                pdY[i] = nullptr;
            }
            if (nullptr != pdX[i]) {
                delete pdX[i];
                pdX[i] = nullptr;
            }

            if (nullptr != pExpandDY[i]) {
                delete pExpandDY[i];
                pExpandDY[i] = nullptr;
            }

        }
        delete[] pdX;
        delete[] pdY;
        delete[] pExpandDY;

    } else {
        // single thread compute
        if (computeW) {
            computeDW(m_pdYTensor, m_pdW[0]);
            computeDb(m_pdYTensor, 0);
        }
        Tensor<float> *pExpandDY = nullptr;
        m_pdYTensor->dilute(pExpandDY, m_tensorSizeBeforeCollapse, m_filterSize, m_stride);
        computeDX(pExpandDY, m_pW[0]);
        if (nullptr != pExpandDY) {
            delete pExpandDY;
            pExpandDY = nullptr;
        }
    }

#endif

}


void TransposedConvolutionLayer::computeDW(const Tensor<float> *pdY, Tensor<float> *pdW) {

    const int N = pdW->getLength();
    Tensor<float> *pExtendX = nullptr;
    if (1 == m_numInputFeatures){
        m_prevLayer->m_pYTensor->dilute(pExtendX, m_prevLayer->m_tensorSize, m_filterSize, m_stride);
    }
    else{
       m_prevLayer->m_pYTensor->dilute(pExtendX, m_prevLayer->m_tensorSize, m_feature_filterSize, m_feature_stride);
    }

    Tensor<float> subX = Tensor<float>(m_tensorSizeBeforeCollapse);
    const int nThreads = (CPUAttr::m_numCPUCore + m_numFilters - 1)/m_numFilters;
    const vector<int> stride1(m_tensorSizeBeforeCollapse.size(),1);
    for (int i = 0; i < N; ++i) {
        pExtendX->subTensorFromTopLeft(pdW->offset2Index(i), &subX, stride1);
        pdW->e(i) += subX.conv(*pdY, nThreads); // + is for batch processing
    }
    if (nullptr != pExtendX) {
        delete pExtendX;
        pExtendX = nullptr;
    }

}


//Note: dx need to accumulate along filters
void
TransposedConvolutionLayer::computeDX(const Tensor<float> *pExpandDY, const Tensor<float> *pW, Tensor<float> *pdX) {

    if (nullptr == pdX) {
        pdX = m_prevLayer->m_pdYTensor;
    }
    const int N = pdX->getLength();
    const int nThread = (CPUAttr::m_numCPUCore+ m_numFilters-1)/m_numFilters;
    const int NRange = (N +nThread -1)/nThread;

    const vector<int> filterSize = (1 == m_numInputFeatures)? m_filterSize : m_feature_filterSize;
    const vector<int> stride = (1 == m_numInputFeatures)? m_stride : m_feature_stride;

    vector<std::thread> threadVec;
    for (int t = 0; t < nThread; ++t) {
        threadVec.push_back(thread(
                [this, t, pExpandDY, pdX, N, pW, NRange, filterSize, stride]() {
                    Tensor<float> subExpandDy = Tensor<float>(filterSize);
                    const vector<int> stride1(filterSize.size(),1);
                    for (int i = NRange * t; i < NRange * (t + 1) && i < N; ++i) {
                        vector<int> index = pdX->offset2Index(i);
                        index = index * stride; // convert to coordinate of extendDY
                        pExpandDY->subTensorFromTopLeft(index, &subExpandDy, stride1);
                        pdX->e(i) += pW->flipConv(subExpandDy);
                    }
                }
        ));

    }
    for (int t = 0; t < threadVec.size(); ++t) {
        threadVec[t].join();
    }

}




