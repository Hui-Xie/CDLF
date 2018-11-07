//
// Created by Hui Xie on 7/19/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "ConvolutionLayer.h"
#include "statisTool.h"
#include <thread>
#include <ConvolutionLayer.h>


ConvolutionLayer::ConvolutionLayer(const int id, const string &name, Layer *prevLayer, const vector<long> &filterSize,
                                    const int numFilters, const int stride)
        : Layer(id, name, {}) {
    if (checkFilterSize(filterSize, prevLayer)) {
        m_type = "ConvolutionLayer";
        m_stride = stride;
        m_filterSize = filterSize;
        m_numFilters = numFilters;
        addPreviousLayer(prevLayer);
        computeOneFiterN();
        updateTensorSize();
        constructFiltersAndY();
    } else {
        cout << "Error: can not construct Convolutional Layer as incorrect Filter Size." << name << endl;
    }

}

ConvolutionLayer::~ConvolutionLayer() {
    //delete Filter Space; the Y space  will delete by base class;
    for (int i = 0; i < m_numFilters; ++i) {
        if (nullptr != m_pW[i]) {
            delete m_pW[i];
            m_pW[i] = nullptr;
        }
        if (nullptr != m_pdW[i]) {
            delete m_pdW[i];
            m_pdW[i] = nullptr;
        }
    }
    delete[] m_pW;
    delete[] m_pdW;
}

// the filterSize in each dimension should be odd,
// or if it is even, it must be same size of corresponding dimension of tensorSize of input X
bool ConvolutionLayer::checkFilterSize(const vector<long> &filterSize, Layer *prevLayer) {
    int dimFilter = filterSize.size();
    int dimX = prevLayer->m_tensorSize.size();
    if (dimFilter == dimX) {
        for (int i = 0; i < dimX; ++i) {
            if (0 == filterSize[i] % 2 && filterSize[i] != prevLayer->m_tensorSize[i]) {
                cout << "Error: the filterSize in each dimension should be odd, "
                        "or if it is even, it must be same size of corresponding dimension of tensorSize of input X."
                     << endl;
                return false;
            }
        }
        return true;
    } else {
        cout << "Error: dimension of filter should be consistent with the output of the previous Layer." << endl;
        return false;
    }
}

void ConvolutionLayer::computeOneFiterN() {
    int N = m_filterSize.size();
    if (N > 0){
        m_OneFilterN = 1;
        for (int i = 0; i < N; ++i) {
            m_OneFilterN *= m_filterSize[i];
        }
    }
    else{
        m_OneFilterN = 0;
    }
}

void ConvolutionLayer::updateTensorSize() {
    m_tensorSize = m_prevLayer->m_tensorSize;
    const int dim = m_tensorSize.size();
    for (int i = 0; i < dim; ++i) {
        m_tensorSize[i] = (m_tensorSize[i] - m_filterSize[i]) / m_stride + 1;
        // ref formula: http://cs231n.github.io/convolutional-networks/
    }

    if (1 != m_numFilters) {
        m_tensorSize.insert(m_tensorSize.begin(), m_numFilters);
    }
    deleteOnes(m_tensorSize);
}

void ConvolutionLayer::constructFiltersAndY() {
    m_pW = (Tensor<float> **) new void *[m_numFilters];
    m_pdW = (Tensor<float> **) new void *[m_numFilters];
    for (int i = 0; i < m_numFilters; ++i) {
        m_pW[i] = new Tensor<float>(m_filterSize);
        m_pdW[i] = new Tensor<float>(m_filterSize);
    }
    allocateYdYTensor();
}


void ConvolutionLayer::initialize(const string &initialMethod) {
    for (int i = 0; i < m_numFilters; ++i) {
        generateGaussian(m_pW[i], 0, sqrt(1.0 / m_OneFilterN));
    }
}

void ConvolutionLayer::zeroParaGradient() {
    for (int i = 0; i < m_numFilters; ++i) {
        m_pdW[i]->zeroInitialize();
    }
}

// Y = W*X
void ConvolutionLayer::forward() {
    const int Nt = m_tensorSize.size();
    const int Nf = m_filterSize.size();
    const int Df = (1 == m_numFilters) ? 0 : 1; //Feature dimension, if number of filter >1, it will add one feature dimension to output Y

    vector<long> f = nonZeroIndex(m_prevLayer->m_tensorSize - m_filterSize);
    Tensor<float> &X = *m_prevLayer->m_pYTensor;

#ifdef Use_GPU
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
    vector<long> Xs(Nf, 0); //the initial topLeft coordinate of subTensor of X
    vector<long> Xc = Xs;  // the topLeft coordinate of subTensor of X
    if (2 == Nt - Df && 1 == f.size()) {
        for (long i = 0; i < m_tensorSize[Df + 0]; ++i) {
            Xc[f[0]] = i * m_stride;
            Tensor<float>* pSubX = nullptr;
            X.subTensorFromTopLeft(Xc, m_filterSize,pSubX);
            if (1 != m_numFilters) {
                /*vector<std::thread> threadVec;
                for (int idxF = 0; idxF < m_numFilters; ++idxF){
                    threadVec.push_back(thread(
                            [this, pSubX, idxF, i](){
                                this->m_pYTensor->e(idxF, i) = pSubX->conv(*(this->m_pW[idxF]));
                            }
                    ));
                }

                for (int t = 0; t < threadVec.size(); ++t){
                    threadVec[t].join();
                }*/

                for(int idxF = 0; idxF < m_numFilters; ++idxF){
                    m_pYTensor->e(idxF, i) = pSubX->conv(*m_pW[idxF]);
                }


            } else {
                m_pYTensor->e(i, 1) = pSubX->conv(*m_pW[0]);  // This maybe has problem.
            }
            if (nullptr != pSubX){
                delete pSubX;
            }
        }
    } else if (2 == Nt - Df && 2 == f.size()) {
        for (long i = 0; i < m_tensorSize[Df + 0]; ++i) {
            Xc[f[0]] = i * m_stride;
            for (long j = 0; j < m_tensorSize[Df + 1]; ++j) {
                Xc[f[1]] = j * m_stride;
                Tensor<float>* pSubX = nullptr;
                X.subTensorFromTopLeft(Xc, m_filterSize,pSubX);
                if (1 != m_numFilters) {
                    /*vector<std::thread> threadVec;
                    for (int idxF = 0; idxF < m_numFilters; ++idxF){
                        threadVec.push_back(thread(
                                [this, pSubX, idxF, i, j](){
                                    this->m_pYTensor->e(idxF, i, j) = pSubX->conv(*(this->m_pW[idxF]));
                                 }
                                                  ));
                    }

                    for (int t = 0; t < threadVec.size(); ++t){
                        threadVec[t].join();
                    }*/

                    for(int idxF = 0; idxF < m_numFilters; ++idxF){
                        m_pYTensor->e(idxF, i,j) = pSubX->conv(*m_pW[idxF]);
                    }

                } else {
                    m_pYTensor->e(i, j) = pSubX->conv(*m_pW[0]);
                }
                if (nullptr != pSubX){
                    delete pSubX;
                }
            }
        }
    } else if (3 == Nt - Df && 3 == f.size()) {
        for (long i = 0; i < m_tensorSize[Df + 0]; ++i) {
            Xc[f[0]] = i * m_stride;
            for (long j = 0; j < m_tensorSize[Df + 1]; ++j) {
                Xc[f[1]] = j * m_stride;
                for (long k = 0; k < m_tensorSize[Df + 2]; ++k) {
                    Xc[f[2]] = k * m_stride;
                    Tensor<float>* pSubX = nullptr;
                    X.subTensorFromTopLeft(Xc, m_filterSize,pSubX);
                    if (1 != m_numFilters) {
                        /*vector<std::thread> threadVec;
                        for (int idxF = 0; idxF < m_numFilters; ++idxF){
                            threadVec.push_back(thread(
                                    [this, pSubX, idxF, i, j, k](){
                                        this->m_pYTensor->e(idxF, i, j, k) = pSubX->conv(*(this->m_pW[idxF]));
                                    }
                            ));
                        }

                        for (int t = 0; t < threadVec.size(); ++t){
                            threadVec[t].join();
                        }*/

                        for(int idxF = 0; idxF < m_numFilters; ++idxF){
                            m_pYTensor->e(idxF, i,j,k) = pSubX->conv(*m_pW[idxF]);
                        }

                    } else {
                        m_pYTensor->e(i, j, k) = pSubX->conv(*m_pW[0]);
                    }
                    if (nullptr != pSubX){
                        delete pSubX;
                    }

                }
            }
        }
    } else if (4 == Nt - Df && 4 == f.size()) {
        for (long i = 0; i < m_tensorSize[Df + 0]; ++i) {
            Xc[f[0]] = i * m_stride;
            for (long j = 0; j < m_tensorSize[Df + 1]; ++j) {
                Xc[f[1]] = j * m_stride;
                for (long k = 0; k < m_tensorSize[Df + 2]; ++k) {
                    Xc[f[2]] = k * m_stride;
                    for (long l = 0; l < m_tensorSize[Df + 3]; ++l) {
                        Xc[f[3]] = l * m_stride;
                        Tensor<float>* pSubX = nullptr;
                        X.subTensorFromTopLeft(Xc, m_filterSize,pSubX);
                        if (1 != m_numFilters) {
                            /*vector<std::thread> threadVec;
                            for (int idxF = 0; idxF < m_numFilters; ++idxF){
                                threadVec.push_back(thread(
                                        [this, pSubX, idxF, i, j, k, l](){
                                            this->m_pYTensor->e(idxF, i, j, k, l) = pSubX->conv(*(this->m_pW[idxF]));
                                        }
                                ));
                            }

                            for (int t = 0; t < threadVec.size(); ++t){
                                threadVec[t].join();
                            }*/

                            for(int idxF = 0; idxF < m_numFilters; ++idxF){
                                m_pYTensor->e(idxF, i,j, k, l) = pSubX->conv(*m_pW[idxF]);
                            }


                        } else {
                            m_pYTensor->e(i, j, k, l) = pSubX->conv(*m_pW[0]);
                        }
                        if (nullptr != pSubX){
                            delete pSubX;
                        }

                    }
                }
            }
        }
    } else if (5 == Nt - Df && 5 == f.size()) {
        for (long i = 0; i < m_tensorSize[Df + 0]; ++i) {
            Xc[f[0]] = i * m_stride;
            for (long j = 0; j < m_tensorSize[Df + 1]; ++j) {
                Xc[f[1]] = j * m_stride;
                for (long k = 0; k < m_tensorSize[Df + 2]; ++k) {
                    Xc[f[2]] = k * m_stride;
                    for (long l = 0; l < m_tensorSize[Df + 3]; ++l) {
                        Xc[f[3]] = l * m_stride;
                        for (long m = 0; m < m_tensorSize[Df + 4]; ++m) {
                            Xc[f[4]] = m * m_stride;
                            Tensor<float>* pSubX = nullptr;
                            X.subTensorFromTopLeft(Xc, m_filterSize,pSubX);
                            if (1 != m_numFilters) {
                               /* vector<std::thread> threadVec;
                                for (int idxF = 0; idxF < m_numFilters; ++idxF){
                                    threadVec.push_back(thread(
                                            [this, pSubX, idxF, i, j, k, l, m](){
                                                this->m_pYTensor->e(idxF, i, j, k, l, m) = pSubX->conv(*(this->m_pW[idxF]));
                                            }
                                    ));
                                }

                                for (int t = 0; t < threadVec.size(); ++t){
                                    threadVec[t].join();
                                }*/

                                for(int idxF = 0; idxF < m_numFilters; ++idxF){
                                    m_pYTensor->e(idxF, i,j, k, l,m) = pSubX->conv(*m_pW[idxF]);
                                }


                            } else {
                                m_pYTensor->e(i, j, k, l, m) = pSubX->conv(*m_pW[0]);
                            }
                            if (nullptr != pSubX){
                                delete pSubX;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (6 == Nt - Df && 6 == f.size()) {
        for (long i = 0; i < m_tensorSize[Df + 0]; ++i) {
            Xc[f[0]] = i * m_stride;
            for (long j = 0; j < m_tensorSize[Df + 1]; ++j) {
                Xc[f[1]] = j * m_stride;
                for (long k = 0; k < m_tensorSize[Df + 2]; ++k) {
                    Xc[f[2]] = k * m_stride;
                    for (long l = 0; l < m_tensorSize[Df + 3]; ++l) {
                        Xc[f[3]] = l * m_stride;
                        for (long m = 0; m < m_tensorSize[Df + 4]; ++m) {
                            Xc[f[4]] = m * m_stride;
                            for (long n = 0; n < m_tensorSize[Df + 5]; ++n) {
                                Xc[f[5]] = n * m_stride;
                                Tensor<float>* pSubX = nullptr;
                                X.subTensorFromTopLeft(Xc, m_filterSize,pSubX);
                                if (1 != m_numFilters) {
                                    /*vector<std::thread> threadVec;
                                    for (int idxF = 0; idxF < m_numFilters; ++idxF){
                                        threadVec.push_back(thread(
                                                [this, pSubX, idxF, i, j, k, l, m, n](){
                                                    this->m_pYTensor->e(idxF, i, j, k, l, m, n) = pSubX->conv(*(this->m_pW[idxF]));
                                                }
                                        ));
                                    }

                                    for (int t = 0; t < threadVec.size(); ++t){
                                        threadVec[t].join();
                                    }*/

                                    for(int idxF = 0; idxF < m_numFilters; ++idxF){
                                        m_pYTensor->e(idxF, i,j, k, l,m,n) = pSubX->conv(*m_pW[idxF]);
                                    }


                                } else {
                                    m_pYTensor->e(i, j, k, l, m, n) = pSubX->conv(*m_pW[0]);
                                }
                                if (nullptr != pSubX){
                                    delete pSubX;
                                }

                            }
                        }
                    }
                }
            }
        }
    }

    else {
        cout << "Error: dimension>6  does not support in convolution forward." << endl;
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

        /*//================== single thread computation =======================
        const vector<long> Xdims = m_prevLayer->m_pYTensor->getDims();
        vector<long> expandDyDims = Xdims + m_filterSize - 1;
        Tensor<float>* pExpandDY = new Tensor<float>(expandDyDims);
        for (int idxF = 0; idxF < m_numFilters; ++idxF) {  //index of Filter
            Tensor<float> *pdY = nullptr;
            m_pdYTensor->extractLowerDTensor(idxF, pdY);
            if (computeW) computeDW(pdY, m_pdW[idxF]);

            expandDyTensor(pdY, pExpandDY);
            computeDX(pExpandDY, m_pW[idxF]);//Note: dx need to accumulate along filters

            if(nullptr != pdY){
                delete pdY;
            }
        }
        if(nullptr != pExpandDY){
            delete pExpandDY;
        }*/

        // ==================multithread computation=======================
        // allocate pdX and pdY along the filters
        Tensor<float>** pdY = (Tensor<float> **) new void *[m_numFilters];
        Tensor<float>** pdX = (Tensor<float> **) new void *[m_numFilters];
        Tensor<float>** pExpandDY = (Tensor<float> **) new void *[m_numFilters];
        for (int i = 0; i < m_numFilters; ++i) {
           pdX[i] = new Tensor<float>(m_prevLayer->m_pdYTensor->getDims());
           pdX[i]->zeroInitialize();  // this is a necessary step as computeDX use += operator
           //pdY memory will be allocated in the extractLowerDTensor function
            const vector<long> Xdims = m_prevLayer->m_pYTensor->getDims();
            vector<long> expandDyDims = Xdims + m_filterSize - 1;
            pExpandDY[i] = new Tensor<float>(expandDyDims);
        }

        vector<std::thread> threadVec;
        for (int idxF = 0; idxF < m_numFilters; ++idxF){
            threadVec.push_back(thread(
                    [this, idxF, pdY, pExpandDY, computeW, pdX](){
                        this->m_pdYTensor->extractLowerDTensor(idxF, pdY[idxF]);
                        if (computeW) this->computeDW(pdY[idxF], this->m_pdW[idxF]);
                        this->expandDyTensor(pdY[idxF], pExpandDY[idxF]);
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

        const vector<long> Xdims = m_prevLayer->m_pYTensor->getDims();
        vector<long> expandDyDims = Xdims + m_filterSize - 1;
        Tensor<float>* pExpandDY = new Tensor<float>(expandDyDims);
        expandDyTensor(m_pdYTensor, pExpandDY);

        computeDX(pExpandDY, m_pW[0]);
        if(nullptr != pExpandDY){
            delete pExpandDY;
        }

    }
}

void ConvolutionLayer::updateParameters(const float lr, const string &method, const int batchSize) {
    if ("sgd" == method) {
        for (int idxF = 0; idxF < m_numFilters; ++idxF) {
            *m_pW[idxF] -= *m_pdW[idxF] * (lr / batchSize);
        }
    }
}


void ConvolutionLayer::expandDyTensor(const Tensor<float> *pdY, Tensor<float>* pExpandDY) {
    //freeExpandDy();
    //const vector<long> Xdims = m_prevLayer->m_pYTensor->getDims();
    //vector<long> expandDyDims = Xdims + m_filterSize - 1;
    //m_expandDy = new Tensor<float>(expandDyDims);
    pExpandDY->zeroInitialize();
    const Tensor<float> &dY = *pdY;
    vector<long> dYTensorSize = pdY->getDims();

    //copy DyTensor to expandDy, according to  m_stride
    int Nt = pdY->getDims().size();  //dyTensor's size, it excludes feature dimension.
    vector<long> Xs = m_filterSize - 1; //X starting coordinates for copying dy
    vector<long> Xc = Xs;

    vector<long> f = nonZeroIndex(m_prevLayer->m_tensorSize - m_filterSize);

    if (2 == Nt && 1 == f.size()) {
        for (long i = 0; i < dYTensorSize[0]; ++i) {
            Xc[f[0]] = Xs[f[0]] + i * m_stride;
            pExpandDY->e(Xc) = dY(i, 0);
        }
    } else if (2 == Nt && 2 == f.size()) {
        for (long i = 0; i < dYTensorSize[0]; ++i) {
            Xc[f[0]] = Xs[f[0]] + i * m_stride;
            for (long j = 0; j < dYTensorSize[1]; ++j) {
                Xc[f[1]] = Xs[f[1]] + j * m_stride;
                pExpandDY->e(Xc) = dY(i, j);
            }
        }
    } else if (3 == Nt && 3 == f.size()) {
        for (long i = 0; i < dYTensorSize[0]; ++i) {
            Xc[f[0]] = Xs[f[0]] + i * m_stride;
            for (long j = 0; j < dYTensorSize[1]; ++j) {
                Xc[f[1]] = Xs[f[1]] + j * m_stride;
                for (long k = 0; k < dYTensorSize[2]; ++k) {
                    Xc[f[2]] = Xs[f[2]] + k * m_stride;
                    pExpandDY->e(Xc) = dY(i, j, k);
                }
            }
        }
    } else if (4 == Nt && 4 == f.size()) {
        for (long i = 0; i < dYTensorSize[0]; ++i) {
            Xc[f[0]] = Xs[f[0]] + i * m_stride;
            for (long j = 0; j < dYTensorSize[1]; ++j) {
                Xc[f[1]] = Xs[f[1]] + j * m_stride;
                for (long k = 0; k < dYTensorSize[2]; ++k) {
                    Xc[f[2]] = Xs[f[2]] + k * m_stride;
                    for (long l = 0; l < dYTensorSize[3]; ++l) {
                        Xc[f[3]] = Xs[f[3]] + l * m_stride;
                        pExpandDY->e(Xc) = dY(i, j, k, l);
                    }
                }
            }
        }
    } else if (5 == Nt && 5 == f.size()) {
        for (long i = 0; i < dYTensorSize[0]; ++i) {
            Xc[f[0]] = Xs[f[0]] + i * m_stride;
            for (long j = 0; j < dYTensorSize[1]; ++j) {
                Xc[f[1]] = Xs[f[1]] + j * m_stride;
                for (long k = 0; k < dYTensorSize[2]; ++k) {
                    Xc[f[2]] = Xs[f[2]] + k * m_stride;
                    for (long l = 0; l < dYTensorSize[3]; ++l) {
                        Xc[f[3]] = Xs[f[3]] + l * m_stride;
                        for (long m = 0; m < dYTensorSize[4]; ++m) {
                            Xc[f[4]] = Xs[f[4]] + m * m_stride;
                            pExpandDY->e(Xc) = dY(i, j, k, l, m);
                        }
                    }
                }
            }
        }
    }else if (6 == Nt && 6 == f.size()) {
        for (long i = 0; i < dYTensorSize[0]; ++i) {
            Xc[f[0]] = Xs[f[0]] + i * m_stride;
            for (long j = 0; j < dYTensorSize[1]; ++j) {
                Xc[f[1]] = Xs[f[1]] + j * m_stride;
                for (long k = 0; k < dYTensorSize[2]; ++k) {
                    Xc[f[2]] = Xs[f[2]] + k * m_stride;
                    for (long l = 0; l < dYTensorSize[3]; ++l) {
                        Xc[f[3]] = Xs[f[3]] + l * m_stride;
                        for (long m = 0; m < dYTensorSize[4]; ++m) {
                            Xc[f[4]] = Xs[f[4]] + m * m_stride;
                            for (long n = 0; n < dYTensorSize[5]; ++n) {
                                Xc[f[5]] = Xs[f[5]] + n * m_stride;
                                pExpandDY->e(Xc) = dY(i, j, k, l, m, n);
                            }
                        }
                    }
                }
            }
        }
    }
    else if (7 == Nt && 7 == f.size()) {
        for (long i = 0; i < dYTensorSize[0]; ++i) {
            Xc[f[0]] = Xs[f[0]] + i * m_stride;
            for (long j = 0; j < dYTensorSize[1]; ++j) {
                Xc[f[1]] = Xs[f[1]] + j * m_stride;
                for (long k = 0; k < dYTensorSize[2]; ++k) {
                    Xc[f[2]] = Xs[f[2]] + k * m_stride;
                    for (long l = 0; l < dYTensorSize[3]; ++l) {
                        Xc[f[3]] = Xs[f[3]] + l * m_stride;
                        for (long m = 0; m < dYTensorSize[4]; ++m) {
                            Xc[f[4]] = Xs[f[4]] + m * m_stride;
                            for (long n = 0; n < dYTensorSize[5]; ++n) {
                                Xc[f[5]] = Xs[f[5]] + n * m_stride;
                                for (long o = 0; o < dYTensorSize[6]; ++o) {
                                    Xc[f[6]] = Xs[f[6]] + o * m_stride;
                                    pExpandDY->e(Xc) = dY(i, j, k, l, m, n, o);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    else {
        cout << "Error: dimension>7  does not support in convolution expandDyTensor." << endl;
    }
}

void ConvolutionLayer::computeDW(const Tensor<float> *pdY, Tensor<float> *pdW) {
    const vector<long> dWDims = pdW->getDims();
    const int Nf = dWDims.size();
    Tensor<float> &X = *m_prevLayer->m_pYTensor;
    const vector<long> dYDims = pdY->getDims();

    vector<long> f = nonZeroIndex(m_prevLayer->m_tensorSize - m_filterSize);
    vector<long> dYDimsEx(Nf, 1); //Nf long integers with each value equal 1
    for (int i = 0; i < dYDims.size() && i < f.size(); ++i) {
        dYDimsEx[f[i]] = dYDims[i];
    }

    if (2 == Nf) {
        for (int i = 0; i < dWDims[0]; ++i) {
            for (int j = 0; j < dWDims[1]; ++j) {
                Tensor<float>* pSubX = nullptr;
                X.subTensorFromTopLeft({i * m_stride, j * m_stride}, dYDimsEx, pSubX, m_stride);
                pdW->e(i, j) += pSubX->conv(*pdY); // + is for batch processing
                if(nullptr != pSubX)
                {
                    delete pSubX;
                }
            }
        }
    } else if (3 == Nf) {
        for (int i = 0; i < dWDims[0]; ++i) {
            for (int j = 0; j < dWDims[1]; ++j) {
                for (int k = 0; k < dWDims[2]; ++k) {
                    Tensor<float>* pSubX = nullptr;
                    X.subTensorFromTopLeft({i * m_stride, j * m_stride, k * m_stride}, dYDimsEx, pSubX, m_stride);
                    pdW->e(i, j, k) += pSubX->conv(*pdY);
                    if(nullptr != pSubX)
                    {
                        delete pSubX;
                    }
                }
            }
        }
    } else if (4 == Nf) {
        for (int i = 0; i < dWDims[0]; ++i) {
            for (int j = 0; j < dWDims[1]; ++j) {
                for (int k = 0; k < dWDims[2]; ++k) {
                    for (int l = 0; l < dWDims[3]; ++l) {
                        Tensor<float>* pSubX = nullptr;
                        X.subTensorFromTopLeft(
                                {i * m_stride, j * m_stride, k * m_stride, l * m_stride}, dYDimsEx, pSubX,m_stride);
                        pdW->e(i, j, k, l) += pSubX->conv(*pdY);
                        if(nullptr != pSubX)
                        {
                            delete pSubX;
                        }
                    }
                }
            }
        }
    } else if (5 == Nf) {
        for (int i = 0; i < dWDims[0]; ++i) {
            for (int j = 0; j < dWDims[1]; ++j) {
                for (int k = 0; k < dWDims[2]; ++k) {
                    for (int l = 0; l < dWDims[3]; ++l) {
                        for (int m = 0; m < dWDims[4]; ++m) {
                            Tensor<float>* pSubX = nullptr;
                            X.subTensorFromTopLeft(
                                    {i * m_stride, j * m_stride, k * m_stride, l * m_stride, m * m_stride}, dYDimsEx, pSubX,m_stride);
                            pdW->e(i, j, k, l, m) += pSubX->conv(*pdY);
                            if(nullptr != pSubX)
                            {
                                delete pSubX;
                            }
                        }
                    }
                }
            }
        }
    } else if (6 == Nf) {
        for (int i = 0; i < dWDims[0]; ++i) {
            for (int j = 0; j < dWDims[1]; ++j) {
                for (int k = 0; k < dWDims[2]; ++k) {
                    for (int l = 0; l < dWDims[3]; ++l) {
                        for (int m = 0; m < dWDims[4]; ++m) {
                            for (int n = 0; n < dWDims[5]; ++n) {
                                Tensor<float>* pSubX = nullptr;
                                X.subTensorFromTopLeft(
                                        {i * m_stride, j * m_stride, k * m_stride, l * m_stride, m * m_stride,
                                         n * m_stride}, dYDimsEx,pSubX, m_stride);
                                pdW->e(i, j, k, l, m, n) += pSubX->conv(*pdY);
                                if(nullptr != pSubX)
                                {
                                    delete pSubX;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    else {
        cout << "Error: Dimension >6 does not support in ConvolutionLayer::computeDW" << endl;
    }
}

//Note: dx need to accumulate along filters
void ConvolutionLayer::computeDX(const Tensor<float> *pExpandDY, const Tensor<float> *pW, Tensor<float>* pdX) {
    if (nullptr == pdX){
        pdX = m_prevLayer->m_pdYTensor;
    }
    const vector<long> dXdims = pdX->getDims();
    const int N = dXdims.size();

    if (2 == N) {
        for (long i = 0; i < dXdims[0]; ++i) {
            for (long j = 0; j < dXdims[1]; ++j) {
                Tensor<float>* pSubExpandDy = nullptr;
                pExpandDY->subTensorFromTopLeft({i, j}, m_filterSize, pSubExpandDy,1);
                pdX->e(i, j) += pSubExpandDy->flip().conv(*pW);
                if (nullptr != pSubExpandDy){
                    delete pSubExpandDy;
                }
            }
        }
    } else if (3 == N) {
        for (long i = 0; i < dXdims[0]; ++i) {
            for (long j = 0; j < dXdims[1]; ++j) {
                for (long k = 0; k < dXdims[2]; ++k) {
                    Tensor<float>* pSubExpandDy = nullptr;
                    pExpandDY->subTensorFromTopLeft({i, j, k}, m_filterSize, pSubExpandDy, 1);
                    pdX->e(i, j, k) += pSubExpandDy->flip().conv(*pW);
                    if (nullptr != pSubExpandDy){
                        delete pSubExpandDy;
                    }
                }
            }
        }
    } else if (4 == N) {
        for (long i = 0; i < dXdims[0]; ++i) {
            for (long j = 0; j < dXdims[1]; ++j) {
                for (long k = 0; k < dXdims[2]; ++k) {
                    for (long l = 0; l < dXdims[3]; ++l) {
                        Tensor<float>* pSubExpandDy = nullptr;
                        pExpandDY->subTensorFromTopLeft({i, j, k, l}, m_filterSize, pSubExpandDy,1);
                        pdX->e(i, j, k, l) += pSubExpandDy->flip().conv(*pW);
                        if (nullptr != pSubExpandDy){
                            delete pSubExpandDy;
                        }
                    }
                }
            }
        }
    } else if (5 == N) {
        for (long i = 0; i < dXdims[0]; ++i) {
            for (long j = 0; j < dXdims[1]; ++j) {
                for (long k = 0; k < dXdims[2]; ++k) {
                    for (long l = 0; l < dXdims[3]; ++l) {
                        for (long m = 0; m < dXdims[4]; ++m) {
                            Tensor<float>* pSubExpandDy = nullptr;
                            pExpandDY->subTensorFromTopLeft({i, j, k, l, m}, m_filterSize, pSubExpandDy, 1);
                            pdX->e(i, j, k, l, m) += pSubExpandDy->flip().conv(*pW);
                            if (nullptr != pSubExpandDy){
                                delete pSubExpandDy;
                            }
                        }
                    }
                }
            }
        }
    } else if (6 == N) {
        for (long i = 0; i < dXdims[0]; ++i) {
            for (long j = 0; j < dXdims[1]; ++j) {
                for (long k = 0; k < dXdims[2]; ++k) {
                    for (long l = 0; l < dXdims[3]; ++l) {
                        for (long m = 0; m < dXdims[4]; ++m) {
                            for (long n = 0; n < dXdims[5]; ++n) {
                                Tensor<float>* pSubExpandDy = nullptr;
                                pExpandDY->subTensorFromTopLeft({i, j, k, l, m, n}, m_filterSize, pSubExpandDy, 1);
                                pdX->e(i, j, k, l, m, n) += pSubExpandDy->flip().conv(*pW);
                                if (nullptr != pSubExpandDy){
                                    delete pSubExpandDy;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else if (7 == N) {
        for (long i = 0; i < dXdims[0]; ++i) {
            for (long j = 0; j < dXdims[1]; ++j) {
                for (long k = 0; k < dXdims[2]; ++k) {
                    for (long l = 0; l < dXdims[3]; ++l) {
                        for (long m = 0; m < dXdims[4]; ++m) {
                            for (long n = 0; n < dXdims[5]; ++n) {
                                for (long o = 0; o < dXdims[6]; ++o) {
                                    Tensor<float>* pSubExpandDy = nullptr;
                                    pExpandDY->subTensorFromTopLeft({i, j, k, l, m, n, o}, m_filterSize, pSubExpandDy, 1);
                                    pdX->e(i, j, k, l, m, n, o) += pSubExpandDy->flip().conv(*pW);
                                    if (nullptr != pSubExpandDy){
                                        delete pSubExpandDy;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        cout << "Error: Dimension >7 does not support in ConvolutionLayer::computeDX." << endl;
    }
}

long ConvolutionLayer::getNumParameters(){
    return m_pW[0]->getLength()*m_numFilters;
}

void ConvolutionLayer::save(const string &netDir) {

}

void ConvolutionLayer::load(const string &netDir) {

}

void ConvolutionLayer::saveStructLine(FILE *pFile) {

}
