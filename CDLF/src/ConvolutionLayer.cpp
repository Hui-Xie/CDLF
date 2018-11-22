//
// Created by Hui Xie on 7/19/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "ConvolutionLayer.h"
#include <thread>


ConvolutionLayer::ConvolutionLayer(const int id, const string &name, Layer *prevLayer, const vector<long> &filterSize,
                                   const int numFilters, const int stride)
        : ConvolutionBasicLayer(id, name, prevLayer, filterSize, numFilters, stride) {
    m_type = "ConvolutionLayer";
    updateTensorSize();
    constructFiltersAndY();
}

ConvolutionLayer::~ConvolutionLayer() {
    //the ConvoltuionBasicLayer is responsible for deleting memory
}


void ConvolutionLayer::updateTensorSize() {
    m_tensorSize = m_prevLayer->m_tensorSize;
    const int dim = m_tensorSize.size();
    for (int i = 0; i < dim; ++i) {
        m_tensorSize[i] = (m_tensorSize[i] - m_filterSize[i]) / m_stride + 1;
        // ref formula: http://cs231n.github.io/convolutional-networks/
    }
    m_tensorSizeBeforeCollapse = m_tensorSize;
    if (1 != m_numFilters) {
        m_tensorSize.insert(m_tensorSize.begin(), m_numFilters);
    }
    deleteOnes(m_tensorSize);
}


// Y = W*X
void ConvolutionLayer::forward() {

#ifdef Use_GPU
    const int Nt = m_tensorSize.size();
    const int Nf = m_filterSize.size();
    const int Df = (1 == m_numFilters) ? 0 : 1; //Feature dimension, if number of filter >1, it will add one feature dimension to output Y

    vector<long> f = nonZeroIndex(m_prevLayer->m_tensorSize - m_filterSize);
    Tensor<float> &X = *m_prevLayer->m_pYTensor;

    long N = length(m_tensorSize)/m_numFilters;
    long NFilter = length(m_filterSize);


    vector<long> filterDimsSpan = dimsSpan(m_filterSize);
    vector<long> yDimsSpan;
    yDimsSpan = m_pYTensor->getDimsSpan();
    if (1 == Df){
        yDimsSpan.erase(yDimsSpan.begin());
    }

    long* pXDimsSpan = nullptr;
    long* pYDimsSpan = nullptr;
    long* pFilterDimsSpan = nullptr;
    long* pNonZeroIndex = nullptr;
    cudaMallocManaged((long**) &pXDimsSpan, X.getDims().size()* sizeof(long));
    cudaMallocManaged((long**) &pYDimsSpan, yDimsSpan.size() * sizeof(long));
    cudaMallocManaged((long**) &pFilterDimsSpan, Nf * sizeof(long));
    cudaMallocManaged((long**) &pNonZeroIndex, f.size() * sizeof(long));
    cudaDeviceSynchronize();

    for(int i=0; i<Nf;++i){
        pXDimsSpan[i] = X.getDimsSpan()[i];
        pFilterDimsSpan[i] = filterDimsSpan[i];
    }

    for(int i=0; i<f.size();++i){
        pYDimsSpan[i] = yDimsSpan[i];
        pNonZeroIndex[i] = f[i];
    }
    //N =1000 is the upper bound for GPU ;
    size_t deviceHeapSize;
    cudaDeviceGetLimit ( &deviceHeapSize, cudaLimitMallocHeapSize);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, deviceHeapSize*1024);
    cudaPrintError();


    for (int idxF=0; idxF<m_numFilters; ++idxF){
        cudaConvLayerForward(X.getData(),pXDimsSpan, m_pW[idxF]->getData(), pFilterDimsSpan, Nf, NFilter,
                             m_stride, m_pYTensor->getData()+idxF*N*sizeof(float), pYDimsSpan, pNonZeroIndex, yDimsSpan.size(), N);
    }
    cudaFree(pXDimsSpan);
    cudaFree(pYDimsSpan);
    cudaFree(pFilterDimsSpan);
    cudaFree(pNonZeroIndex);

#else
    long N = length(m_tensorSize) / m_numFilters;
    vector<long> dimsSpanBeforeCollpase = genDimsSpan(m_tensorSizeBeforeCollapse);
    Tensor<float> **pSubX = (Tensor<float> **) new void *[m_numFilters];

    vector<std::thread> threadVec;
    for (int idxF = 0; idxF < m_numFilters; ++idxF) {
        threadVec.push_back(thread(
                [this, idxF, pSubX, N, &dimsSpanBeforeCollpase]() {
                    pSubX[idxF] = new Tensor<float>(m_filterSize);
                    for (long i = 0; i < N; ++i) {
                        vector<long> index = m_pYTensor->offset2Index(dimsSpanBeforeCollpase, i);
                        m_prevLayer->m_pYTensor->subTensorFromTopLeft(index * m_stride, pSubX[idxF]);
                        m_pYTensor->e(i + idxF * N) = pSubX[idxF]->conv(*m_pW[idxF]);
                    }
                    if (nullptr != pSubX[idxF]) {
                        delete pSubX[idxF];
                        pSubX[idxF] = nullptr;
                    }
                }
        ));
    }
    for (int t = 0; t < threadVec.size(); ++t) {
        threadVec[t].join();
    }
    if (nullptr != pSubX) {
        delete[] pSubX;
        pSubX = nullptr;
    }
#endif


}

// Y =W*X
// dL/dW = dL/dY * dY/dW;
// dL/dX = dL/dY * dY/dX;
// algorithm ref: https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
void ConvolutionLayer::backward(bool computeW) {
    // dX needs to consider the accumulation of different filters
    if (1 != m_numFilters) {
        // ==================multithread computation=======================
        // allocate pdX and pdY along the filters
        Tensor<float> **pdY = (Tensor<float> **) new void *[m_numFilters];
        Tensor<float> **pdX = (Tensor<float> **) new void *[m_numFilters];
        Tensor<float> **pExpandDY = (Tensor<float> **) new void *[m_numFilters];
        for (int i = 0; i < m_numFilters; ++i) {
            pdX[i] = new Tensor<float>(m_prevLayer->m_pdYTensor->getDims());
            pdX[i]->zeroInitialize();  // this is a necessary step as computeDX use += operator
            //pdY memory will be allocated in the extractLowerDTensor function
            //pExpandDY memory will be allocated in the dilute method;
        }

        vector<std::thread> threadVec;
        for (int idxF = 0; idxF < m_numFilters; ++idxF) {
            threadVec.push_back(thread(
                    [this, idxF, pdY, pExpandDY, computeW, pdX]() {
                        this->m_pdYTensor->extractLowerDTensor(idxF, pdY[idxF]);
                        if (computeW) this->computeDW(pdY[idxF], this->m_pdW[idxF]);
                        pdY[idxF]->dilute(pExpandDY[idxF], m_tensorSizeBeforeCollapse, m_filterSize - 1, m_stride);
                        this->computeDX(pExpandDY[idxF], this->m_pW[idxF],
                                        pdX[idxF]); //as pdX needs to accumulate, pass pointer
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
        if (computeW) computeDW(m_pdYTensor, m_pdW[0]);
        Tensor<float> *pExpandDY = nullptr;
        m_pdYTensor->dilute(pExpandDY, m_tensorSizeBeforeCollapse, m_filterSize - 1, m_stride);
        computeDX(pExpandDY, m_pW[0]);
        if (nullptr != pExpandDY) {
            delete pExpandDY;
            pExpandDY = nullptr;
        }

    }
}


void ConvolutionLayer::computeDW(const Tensor<float> *pdY, Tensor<float> *pdW) {
    long N = pdW->getLength();
    // compute proper number of threads
    int nThread = (CPUAttr::m_numCPUCore * 30 - m_numFilters - 1) / m_numFilters;
    if (nThread < 1) {
        nThread = 1;
    }
    if (nThread > N) {
        nThread = N;
    }
    Tensor<float> **pSubX = (Tensor<float> **) new void *[nThread];
    const long range = (N + nThread - 1) / nThread;

    vector<std::thread> threadVec;
    for (int t = 0; t < nThread; ++t) {
        threadVec.push_back(thread(
                [this, t, range, pSubX, N, pdW, pdY]() {
                    pSubX[t] = new Tensor<float>(m_tensorSizeBeforeCollapse);
                    for (long i = range * t; i < range * (t + 1) && i < N; ++i) {
                        m_prevLayer->m_pYTensor->subTensorFromTopLeft(pdW->offset2Index(i) * m_stride, pSubX[t],m_stride);
                        pdW->e(i) += pSubX[t]->conv(*pdY); // + is for batch processing
                    }
                    if (nullptr != pSubX[t]) {
                        delete pSubX[t];
                        pSubX[t] = nullptr;
                    }
                }
        ));
    }
    for (int t = 0; t < threadVec.size(); ++t) {
        threadVec[t].join();
    }

    if (nullptr != pSubX) {
        delete[] pSubX;
        pSubX = nullptr;
    }
}

//Note: dx need to accumulate along filters
void ConvolutionLayer::computeDX(const Tensor<float> *pExpandDY, const Tensor<float> *pW, Tensor<float> *pdX) {
    if (nullptr == pdX) {
        pdX = m_prevLayer->m_pdYTensor;
    }
    long N = pdX->getLength();

    // compute proper number of threads
    int nThread = (CPUAttr::m_numCPUCore * 30 - m_numFilters - 1) / m_numFilters;
    if (nThread < 10) {
        nThread = 10;
    }
    if (nThread > N) {
        nThread = N;
    }
    Tensor<float> **pSubExpandDy = (Tensor<float> **) new void *[nThread];
    const long range = (N + nThread - 1) / nThread;

    vector<std::thread> threadVec;
    for (int t = 0; t < nThread; ++t) {
        threadVec.push_back(thread(
                [this, t, range, pExpandDY, pSubExpandDy, pdX, N, pW]() {
                    pSubExpandDy[t] = new Tensor<float>(m_filterSize);
                    for (long i = range * t; i < range * (t + 1) && i < N; ++i) {
                        vector<long> index = pdX->offset2Index(i);
                        pExpandDY->subTensorFromTopLeft(index, pSubExpandDy[t], 1);
                        pdX->e(i) += pSubExpandDy[t]->flip().conv(*pW);
                    }
                    if (nullptr != pSubExpandDy[t]) {
                        delete pSubExpandDy[t];
                        pSubExpandDy[t] = nullptr;
                    }
                }
        ));

    }
    for (int t = 0; t < threadVec.size(); ++t) {
        threadVec[t].join();
    }
    if (nullptr != pSubExpandDy) {
        delete[] pSubExpandDy;
        pSubExpandDy = nullptr;
    }
}
