//
// Created by Hui Xie on 7/28/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "MaxPoolingLayer.h"


MaxPoolingLayer::MaxPoolingLayer(const int id, const string &name, Layer *prevLayer, const vector<long> &filterSize,
                                  const int stride)
        : Layer(id, name, {}) {

    m_type = "MaxPoolingLayer";
    m_stride = stride;
    m_filterSize = filterSize;
    m_tensorSize = prevLayer->m_tensorSize; // this is initial, not final size

    int N = filterSize.size();
    m_OneFilterN = 1;
    for (int i = 0; i < N; ++i) {
        m_OneFilterN *= filterSize[i];
    }
    addPreviousLayer(prevLayer);
    constructY();
}

MaxPoolingLayer::~MaxPoolingLayer() {
    //null
}

void MaxPoolingLayer::constructY() {
    //get refined pYTensor size
    const int dim = m_tensorSize.size();
    for (int i = 0; i < dim; ++i) {
        m_tensorSize[i] = (m_tensorSize[i] - m_filterSize[i]) / m_stride + 1;
        // ref formula: http://cs231n.github.io/convolutional-networks/
    }

    if (0 != m_tensorSize.size()) {
        m_pYTensor = new Tensor<float>(m_tensorSize);
        m_pdYTensor = new Tensor<float>(m_tensorSize);
    } else {
        m_pYTensor = nullptr;
        m_pdYTensor = nullptr;
    }
}

void MaxPoolingLayer::initialize(const string &initialMethod) {
    //null
}

void MaxPoolingLayer::zeroParaGradient() {
    //null
}

// Y_i = max(X_i) in filterSize range
void MaxPoolingLayer::forward() {
    const int N = m_filterSize.size();
    Tensor<float> &X = *m_prevLayer->m_pYTensor;
    if (2 == N) {
        vector<long> Xc = {0, 0}; // start point
        for (long i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] = i * m_stride;
            for (long j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] = j * m_stride;
                Tensor<float>* pSubX = nullptr;
                X.subTensorFromTopLeft(Xc, m_filterSize, pSubX);
                m_pYTensor->e(i, j) = pSubX->max();
                if (nullptr != pSubX){
                    delete pSubX;
                }
            }
        }
    } else if (3 == N) {
        vector<long> Xc = {0, 0, 0};
        for (long i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] = i * m_stride;
            for (long j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] = j * m_stride;
                for (long k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] = k * m_stride;
                    Tensor<float>* pSubX = nullptr;
                    X.subTensorFromTopLeft(Xc, m_filterSize, pSubX);
                    m_pYTensor->e(i, j, k) = pSubX->max();
                    if (nullptr != pSubX){
                        delete pSubX;
                    }
                }
            }
        }
    } else if (4 == N) {
        vector<long> Xc = {0, 0, 0, 0};
        for (long i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] = i * m_stride;
            for (long j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] = j * m_stride;
                for (long k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] = k * m_stride;
                    for (long l = 0; l < m_tensorSize[3]; ++l) {
                        Xc[3] = l * m_stride;
                        Tensor<float>* pSubX = nullptr;
                        X.subTensorFromTopLeft(Xc, m_filterSize, pSubX);
                        m_pYTensor->e(i, j, k, l) = pSubX->max();
                        if (nullptr != pSubX){
                            delete pSubX;
                        }
                    }
                }
            }
        }
    } else if (5 == N) {
        vector<long> Xc = {0, 0, 0, 0, 0};
        for (long i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] = i * m_stride;
            for (long j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] = j * m_stride;
                for (long k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] = k * m_stride;
                    for (long l = 0; l < m_tensorSize[3]; ++l) {
                        Xc[3] = l * m_stride;
                        for (long m = 0; m < m_tensorSize[4]; ++m) {
                            Xc[4] = m * m_stride;
                            Tensor<float>* pSubX = nullptr;
                            X.subTensorFromTopLeft(Xc, m_filterSize, pSubX);
                            m_pYTensor->e(i, j, k, l, m) = pSubX->max();
                            if (nullptr != pSubX){
                                delete pSubX;
                            }
                        }
                    }
                }
            }
        }
    } else if (6 == N) {
        vector<long> Xc = {0, 0, 0, 0, 0, 0};
        for (long i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] = i * m_stride;
            for (long j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] = j * m_stride;
                for (long k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] = k * m_stride;
                    for (long l = 0; l < m_tensorSize[3]; ++l) {
                        Xc[3] = l * m_stride;
                        for (long m = 0; m < m_tensorSize[4]; ++m) {
                            Xc[4] = m * m_stride;
                            for (long o = 0; o < m_tensorSize[4]; ++o) {
                                Xc[5] = o * m_stride;
                                Tensor<float>* pSubX = nullptr;
                                X.subTensorFromTopLeft(Xc, m_filterSize, pSubX);
                                m_pYTensor->e(i, j, k, l, m, o) = pSubX->max();
                                if (nullptr != pSubX){
                                    delete pSubX;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        cout << "Error: dimension>6  does not support in MaxPooling Layer forward." << endl;
    }
}

// Y_i = max(X_i) in filterSize range
// dL/dX_i = dL/dY * 1 when Xi = max; 0 otherwise;
void MaxPoolingLayer::backward(bool computeW) {
    Tensor<float> &dLdy = *m_pdYTensor;
    const int N = m_filterSize.size();
    Tensor<float> &dX = *(m_prevLayer->m_pdYTensor);
    Tensor<float> &X = *(m_prevLayer->m_pYTensor);
    if (2 == N) {
        vector<long> Xc = {0, 0};
        for (long i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] = i * m_stride;
            for (long j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] = j * m_stride;
                Tensor<float>* pSubX = nullptr;
                X.subTensorFromTopLeft(Xc, m_filterSize,pSubX);
                const float maxValue = pSubX->max();
                for (long ii = 0; ii < m_filterSize[0]; ++ii)
                    for (long jj = 0; jj < m_filterSize[1]; ++jj)
                        if (maxValue == X(Xc[0] + ii, Xc[1] + jj)) dX(Xc[0] + ii, Xc[1] + jj) += dLdy(i, j);
                if (nullptr != pSubX)
                {
                    delete pSubX;
                }
            }
        }
    } else if (3 == N) {
        vector<long> Xc = {0, 0, 0};
        for (long i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] = i * m_stride;
            for (long j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] = j * m_stride;
                for (long k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] = k * m_stride;
                    Tensor<float>* pSubX = nullptr;
                    X.subTensorFromTopLeft(Xc, m_filterSize,pSubX);
                    const float maxValue = pSubX->max();
                    for (long ii = 0; ii < m_filterSize[0]; ++ii)
                        for (long jj = 0; jj < m_filterSize[1]; ++jj)
                            for (long kk = 0; kk < m_filterSize[2]; ++kk)
                                if (maxValue == X(Xc[0] + ii, Xc[1] + jj, Xc[2] + kk))
                                    dX(Xc[0] + ii, Xc[1] + jj, Xc[2] + kk) += dLdy(i, j, k);
                    if (nullptr != pSubX)
                    {
                        delete pSubX;
                    }
                }
            }
        }
    } else if (4 == N) {
        vector<long> Xc = {0, 0, 0, 0};
        for (long i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] = i * m_stride;
            for (long j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] = j * m_stride;
                for (long k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] = k * m_stride;
                    for (long l = 0; l < m_tensorSize[3]; ++l) {
                        Xc[3] = l * m_stride;
                        Tensor<float>* pSubX = nullptr;
                        X.subTensorFromTopLeft(Xc, m_filterSize,pSubX);
                        const float maxValue = pSubX->max();
                        for (long ii = 0; ii < m_filterSize[0]; ++ii)
                            for (long jj = 0; jj < m_filterSize[1]; ++jj)
                                for (long kk = 0; kk < m_filterSize[2]; ++kk)
                                    for (long ll = 0; ll < m_filterSize[3]; ++ll)
                                        if (maxValue == X(Xc[0] + ii, Xc[1] + jj, Xc[2] + kk, Xc[3] + ll))
                                            dX(Xc[0] + ii, Xc[1] + jj, Xc[2] + kk, Xc[3] + ll) += dLdy(i, j, k, l);
                        if (nullptr != pSubX)
                        {
                            delete pSubX;
                        }
                    }
                }
            }
        }
    } else if (5 == N) {
        vector<long> Xc = {0, 0, 0, 0, 0};
        for (long i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] = i * m_stride;
            for (long j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] = j * m_stride;
                for (long k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] = k * m_stride;
                    for (long l = 0; l < m_tensorSize[3]; ++l) {
                        Xc[3] = l * m_stride;
                        for (long m = 0; m < m_tensorSize[4]; ++m) {
                            Xc[4] = m * m_stride;
                            Tensor<float>* pSubX = nullptr;
                            X.subTensorFromTopLeft(Xc, m_filterSize,pSubX);
                            const float maxValue = pSubX->max();
                            for (long ii = 0; ii < m_filterSize[0]; ++ii)
                                for (long jj = 0; jj < m_filterSize[1]; ++jj)
                                    for (long kk = 0; kk < m_filterSize[2]; ++kk)
                                        for (long ll = 0; ll < m_filterSize[3]; ++ll)
                                            for (long mm = 0; mm < m_filterSize[4]; ++mm)
                                                if (maxValue ==
                                                    X(Xc[0] + ii, Xc[1] + jj, Xc[2] + kk, Xc[3] + ll, Xc[4] + mm))
                                                    dX(Xc[0] + ii, Xc[1] + jj, Xc[2] + kk, Xc[3] + ll,
                                                       Xc[4] + mm) += dLdy(i, j, k, l, m);
                            if (nullptr != pSubX)
                            {
                                delete pSubX;
                            }
                        }
                    }
                }
            }
        }
    }  else if (6 == N) {
        vector<long> Xc = {0, 0, 0, 0, 0, 0};
        for (long i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] = i * m_stride;
            for (long j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] = j * m_stride;
                for (long k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] = k * m_stride;
                    for (long l = 0; l < m_tensorSize[3]; ++l) {
                        Xc[3] = l * m_stride;
                        for (long m = 0; m < m_tensorSize[4]; ++m) {
                            Xc[4] = m * m_stride;
                            for (long o = 0; o < m_tensorSize[5]; ++o) {
                                Xc[5] = o * m_stride;
                                Tensor<float>* pSubX = nullptr;
                                X.subTensorFromTopLeft(Xc, m_filterSize,pSubX);
                                const float maxValue = pSubX->max();
                                for (long ii = 0; ii < m_filterSize[0]; ++ii)
                                    for (long jj = 0; jj < m_filterSize[1]; ++jj)
                                        for (long kk = 0; kk < m_filterSize[2]; ++kk)
                                            for (long ll = 0; ll < m_filterSize[3]; ++ll)
                                                for (long mm = 0; mm < m_filterSize[4]; ++mm)
                                                    for (long oo = 0; oo < m_filterSize[5]; ++oo)
                                                    if (maxValue ==
                                                        X(Xc[0] + ii, Xc[1] + jj, Xc[2] + kk, Xc[3] + ll, Xc[4] + mm, Xc[5]+oo))
                                                        dX(Xc[0] + ii, Xc[1] + jj, Xc[2] + kk, Xc[3] + ll,
                                                           Xc[4] + mm, Xc[5]+oo ) += dLdy(i, j, k, l, m, o);
                                if (nullptr != pSubX)
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
        cout << "Error: dimension>6  does not support in MaxPooling Layer backward." << endl;
    }
}

void MaxPoolingLayer::updateParameters(const float lr, const string &method, const int batchSize) {
    //null
}


