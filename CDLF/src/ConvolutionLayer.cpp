//
// Created by Hui Xie on 7/19/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "ConvolutionLayer.h"
#include <thread>
#ifdef Use_GPU
   #include "CudnnConvolution.h"
#endif

ConvolutionLayer::ConvolutionLayer(const int id, const string &name, Layer *prevLayer, const vector<int> &filterSize,
                                   const vector<int>& stride, const int numFilters)
        : ConvolutionBasicLayer(id, name, prevLayer, filterSize, stride, numFilters) {
    m_type = "ConvolutionLayer";
    if (!isElementBiggerThan0(stride)){
        cout<<"Error: the stride of convolutionLayer should be greater than 0. "<<endl;
    }
    updateTensorSize();
    constructFiltersAndY();
}

ConvolutionLayer::~ConvolutionLayer() {
    //the ConvoltuionBasicLayer is responsible for deleting memory
}


void ConvolutionLayer::updateTensorSize() {
    m_tensorSize = m_prevLayer->m_tensorSize;
    const int dim = m_tensorSize.size();

    const int s = (1 == m_prevLayer->m_numFeatures)? 0: 1; //indicate whether previousTensor includes feature dimension

    for (int i = 0; i < dim; ++i) {
        m_tensorSize[i+s] = (m_tensorSize[i+s] - m_filterSize[i]) / m_stride[i] + 1;
        // ref formula: http://cs231n.github.io/convolutional-networks/
    }

    if (0 ==s){
        m_tensorSize.insert(m_tensorSize.begin(), 1);
    }
    else{
        m_tensorSize[0] = 1;
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
    CudnnConvolution cudnnConvolution(this, m_filterSize, m_stride, m_numFilters);
    cudnnConvolution.forward();

#else
    cout<<"Convolution need GPU."<<endl;
    /*
    const int N = length(m_tensorSize) / m_numFilters;
    const vector<int> dimsSpanBeforeCollpase = genDimsSpan(m_tensorSizeBeforeCollapse);
    const int nThread = (CPUAttr::m_numCPUCore+ m_numFilters-1)/m_numFilters;
    const int NRange = (N +nThread -1)/nThread;

    vector<std::thread> threadVec;
    for (int idxF = 0; idxF < m_numFilters; ++idxF) {
        for (int t = 0; t < nThread; ++t) {
            threadVec.push_back(thread(
                    [this, idxF, t, N, &dimsSpanBeforeCollpase, NRange]() {
                        Tensor<float> subX = Tensor<float>(m_filterSize);
                        const int offseti = idxF * N;
                        const vector<int> stride1(m_filterSize.size(),1);
                        for (int i = NRange*t; i<NRange*(t+1) && i < N; ++i) {
                            m_prevLayer->m_pYTensor->subTensorFromTopLeft(
                                    m_pYTensor->offset2Index(dimsSpanBeforeCollpase, i) * m_stride, &subX, stride1);
                            m_pYTensor->e(offseti+i) = subX.conv(*m_pW[idxF]);
                        }
                    }
            ));
        }
    }
    for (int t = 0; t < threadVec.size(); ++t) {
        threadVec[t].join();
    }
    */
#endif

}

// Y =W*X
// dL/dW = dL/dY * dY/dW;
// dL/dX = dL/dY * dY/dX;
// algorithm ref: https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
void ConvolutionLayer::backward(bool computeW, bool computeX) {

#ifdef Use_GPU
    CudnnConvolution cudnnConvolution(this, m_filterSize, m_stride, m_numFilters);
    cudnnConvolution.backward(computeW, computeX);
#else

    cout<<"Convolution need GPU."<<endl;
    /*
    // dX needs to consider the accumulation of different filters
    if (1 != m_numFilters) {
        //==============Single Thread computation==========================
        // single thread is a little slower than multithread, but save memory.

//        for (int idxF = 0; idxF < m_numFilters; ++idxF){
//            Tensor<float>* pdY = nullptr;
//            m_pdYTensor->extractLowerDTensor(idxF, pdY);
//            if (computeW) computeDW(pdY, m_pdW[idxF]);
//            Tensor<float> *pExpandDY = nullptr;
//            pdY->dilute(pExpandDY, m_tensorSizeBeforeCollapse, m_filterSize - 1, m_stride);
//            computeX(pExpandDY, m_pW[idxF]);
//            if (nullptr != pExpandDY) {
//                delete pExpandDY;
//                pExpandDY = nullptr;
//            }
//            if (nullptr != pdY) {
//                delete pdY;
//                pdY = nullptr;
//            }
//        }


        // ==================multithread computation=======================
        // allocate pdX and pdY along the filters
        Tensor<float> **pdX = (Tensor<float> **) new void *[m_numFilters];
        Tensor<float> **pdY = (Tensor<float> **) new void *[m_numFilters];
        Tensor<float> **pExpandDY = (Tensor<float> **) new void *[m_numFilters];
        for (int i = 0; i < m_numFilters; ++i) {
            pdX[i] = new Tensor<float>(m_prevLayer->m_tensorSize);
            pdX[i]->zeroInitialize();  // this is a necessary step as computeX use += operator
            pdY[i] = nullptr; //pdY memory will be allocated in the extractLowerDTensor function
            pExpandDY[i] = nullptr; //pExpandDY memory will be allocated in the dilute method;
        }

        vector<std::thread> threadVec;
        for (int idxF = 0; idxF < m_numFilters; ++idxF) {
            threadVec.push_back(thread(
                    [this, idxF, pdY, pExpandDY, computeW, computeX, pdX]() {
                        this->m_pdYTensor->extractLowerDTensor(idxF, pdY[idxF]);
                        if (computeW) this->computeDW(pdY[idxF], this->m_pdW[idxF]);
                        if (computeX){
                            pdY[idxF]->dilute(pExpandDY[idxF], m_tensorSizeBeforeCollapse, m_filterSize, m_stride);
                            this->computeDX(pExpandDY[idxF], this->m_pW[idxF], pdX[idxF]); //as pdX needs to accumulate, pass pointer
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
        if (computeW) computeDW(m_pdYTensor, m_pdW[0]);
        Tensor<float> *pExpandDY = nullptr;
        m_pdYTensor->dilute(pExpandDY, m_tensorSizeBeforeCollapse, m_filterSize, m_stride);
        computeDX(pExpandDY, m_pW[0]);
        if (nullptr != pExpandDY) {
            delete pExpandDY;
            pExpandDY = nullptr;
        }

    }
*/
#endif

}


void ConvolutionLayer::computeDW(const Tensor<float> *pdY, Tensor<float> *pdW) {

   /*
    * const int N = pdW->getLength();  // the N of DW is small, it does not need thread.
    Tensor<float> subX = Tensor<float>(m_tensorSizeBeforeCollapse);
    const int nThreads = (CPUAttr::m_numCPUCore + m_numFilters - 1)/m_numFilters;
    for (int i = 0; i < N; ++i) {
        m_prevLayer->m_pYTensor->subTensorFromTopLeft(pdW->offset2Index(i), &subX, m_stride);
        pdW->e(i) += subX.conv(*pdY, nThreads); // + is for batch processing
    }
    */
}

//Note: dx need to accumulate along filters
void ConvolutionLayer::computeDX(const Tensor<float> *pExpandDY, const Tensor<float> *pW, Tensor<float> *pdX) {

    /*
    if (nullptr == pdX) {
        pdX = m_prevLayer->m_pdYTensor;
    }
    const int N = pdX->getLength();
    const int nThread = (CPUAttr::m_numCPUCore+ m_numFilters-1)/m_numFilters;
    const int NRange = (N +nThread -1)/nThread;

    vector<std::thread> threadVec;
    for (int t = 0; t < nThread; ++t) {
        threadVec.push_back(thread(
                [this, t, pExpandDY, pdX, N, pW, NRange]() {
                    Tensor<float> subExpandDy = Tensor<float>(m_filterSize);
                    const vector<int> stride1(m_filterSize.size(),1);
                    for (int i = NRange * t; i < NRange * (t + 1) && i < N; ++i) {
                        pExpandDY->subTensorFromTopLeft(pdX->offset2Index(i), &subExpandDy, stride1);
                        pdX->e(i) += pW->flipConv(subExpandDy);
                    }
                }
        ));

    }
    for (int t = 0; t < threadVec.size(); ++t) {
        threadVec[t].join();
    }
    */
}
