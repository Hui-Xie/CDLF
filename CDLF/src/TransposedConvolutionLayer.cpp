//
// Created by Hui Xie on 11/19/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "TransposedConvolutionLayer.h"
#include <thread>

TransposedConvolutionLayer::TransposedConvolutionLayer(const int id, const string &name, Layer *prevLayer, const vector<long> &filterSize,
                                   const int numFilters, const int stride)
        : ConvolutionBasicLayer(id, name, prevLayer, filterSize, numFilters, stride)
{
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
    const int dim = m_prevLayer->m_tensorSize.size();
    for (int i = 0; i < dim; ++i) {
        m_tensorSize[i] = (m_prevLayer->m_tensorSize[i] - 1) * m_stride + m_filterSize[i];
    }
    m_tensorSizeBeforeCollapse = m_tensorSize;
    if (1 != m_numFilters) {
        m_tensorSize.insert(m_tensorSize.begin(), m_numFilters);
    }
    deleteOnes(m_tensorSize);
}


// Y = W*X
void TransposedConvolutionLayer::forward() {

#ifdef Use_GPU //need to modify for TransposedConvolutionLayer
    long N = length(m_tensorSize)/m_numFilters;
    long NFilter = length(m_filterSize);


    vector<long> filterDimsSpan = dimsSpan(m_filterSize);
    vector<long> yDimsSpan;
    yDimsSpan = m_pYTensor->getDimsSpan();
    if (1 != m_numFilters){
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
    Tensor<float>* pExtendX = nullptr;
    m_prevLayer->m_pYTensor->dilute(pExtendX, m_prevLayer->m_pYTensor->getDims(),  m_filterSize-1, m_stride);
    Tensor<float> *pSubX = new Tensor<float>(m_filterSize);
    for (int idxF = 0; idxF < m_numFilters; ++idxF) {
        for (long i = 0; i < N; ++i) {
            vector<long> index = m_pYTensor->offset2Index(dimsSpanBeforeCollpase, i);
            pExtendX->subTensorFromTopLeft(index * m_stride, pSubX);
            m_pYTensor->e(i + idxF * N) = pSubX->conv(*m_pW[idxF]);
        }
    }
    if (nullptr != pSubX) {
        delete pSubX;
        pSubX = nullptr;
    }

    if(nullptr != pExtendX){
        delete pExtendX;
        pExtendX = nullptr;
    }
#endif


}

// Y =W*X
// dL/dW = dL/dY * dY/dW;
// dL/dX = dL/dY * dY/dX;
// algorithm ref: https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
void TransposedConvolutionLayer::backward(bool computeW) {
    // dX needs to consider the accumulation of different filters
    if (1 != m_numFilters) {
        // ==================multithread computation=======================
        // allocate pdX and pdY along the filters
        Tensor<float>** pdY = (Tensor<float> **) new void *[m_numFilters];
        Tensor<float>** pdX = (Tensor<float> **) new void *[m_numFilters];
        Tensor<float>** pExpandDY = (Tensor<float> **) new void *[m_numFilters];
        for (int i = 0; i < m_numFilters; ++i) {
            pdX[i] = new Tensor<float>(m_prevLayer->m_pdYTensor->getDims());
            pdX[i]->zeroInitialize();  // this is a necessary step as computeDX use += operator
            //pdY memory will be allocated in the extractLowerDTensor function
            //pExpandDY memory will be allocated in the dilute method;
        }
        vector<std::thread> threadVec;
        for (int idxF = 0; idxF < m_numFilters; ++idxF){
            threadVec.push_back(thread(
                    [this, idxF, pdY, pExpandDY, computeW, pdX](){
                        this->m_pdYTensor->extractLowerDTensor(idxF, pdY[idxF]);
                        if (computeW) this->computeDW(pdY[idxF], this->m_pdW[idxF]);
                        pdY[idxF]->dilute(pExpandDY[idxF], m_tensorSizeBeforeCollapse, m_filterSize-1, 1);
                        this->computeDX(pExpandDY[idxF], this->m_pW[idxF], pdX[idxF]); //as pdX needs to accumulate, pass pointer
                    }
            ));
        }

        for (int t = 0; t < threadVec.size(); ++t){
            threadVec[t].join();
        }

        // accumulate pdX
        for (int idxF = 0; idxF < m_numFilters; ++idxF){
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
        if (computeW)  computeDW(m_pdYTensor, m_pdW[0]);
        Tensor<float>* pExpandDY = nullptr;
        m_pdYTensor->dilute(pExpandDY, m_tensorSizeBeforeCollapse, m_filterSize-1, 1);
        computeDX(pExpandDY, m_pW[0]);
        if(nullptr != pExpandDY){
            delete pExpandDY;
            pExpandDY = nullptr;
        }
    }
}


void TransposedConvolutionLayer::computeDW(const Tensor<float> *pdY, Tensor<float> *pdW) {
    Tensor<float>* pExtendX = nullptr;
    m_prevLayer->m_pYTensor->dilute(pExtendX, m_prevLayer->m_pYTensor->getDims(), m_filterSize-1, m_stride);
    Tensor<float>* pSubX = new Tensor<float>(m_tensorSizeBeforeCollapse);
    long N = pdW->getLength();
    for (long i=0; i<N; ++i){
        pExtendX->subTensorFromTopLeft(pdW->offset2Index(i), pSubX, 1);
        pdW->e(i) += pSubX->conv(*pdY); // + is for batch processing
    }

    if(nullptr != pSubX)
    {
        delete pSubX;
        pSubX = nullptr;
    }

    if (nullptr != pExtendX){
        delete pExtendX;
        pExtendX = nullptr;
    }
}

//Note: dx need to accumulate along filters
void TransposedConvolutionLayer::computeDX(const Tensor<float> *pExpandDY, const Tensor<float> *pW, Tensor<float>* pdX) {
    if (nullptr == pdX){
        pdX = m_prevLayer->m_pdYTensor;
    }
    Tensor<float>* pSubExpandDy = new Tensor<float>(m_filterSize);
    long N = pdX->getLength();
    for(long i=0; i< N; ++i){
        vector<long> index = pdX->offset2Index(i);
        index = index*m_stride; // convert to coordinate of extendDY
        pExpandDY->subTensorFromTopLeft(index, pSubExpandDy,1);
        pdX->e(i) += pSubExpandDy->flip().conv(*pW);
    }
    if (nullptr != pSubExpandDy){
        delete pSubExpandDy;
        pSubExpandDy = nullptr;
    }
}


