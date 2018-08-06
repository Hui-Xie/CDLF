//
// Created by Hui Xie on 7/28/2018.
//

#include "MaxPoolingLayer.h"


MaxPoolingLayer::MaxPoolingLayer(const int id, const string &name, const vector<int> &filterSize,
                                 Layer *prevLayer, const int stride)
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
        vector<int> Xc = {0, 0};
        for (int i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] += i * m_stride;
            for (int j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] += j * m_stride;
                m_pYTensor->e(i, j) = X.subTensorFromTopLeft(Xc, m_filterSize).max();
            }
        }
    } else if (3 == N) {
        vector<int> Xc = {0, 0, 0};
        for (int i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] += i * m_stride;
            for (int j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] += j * m_stride;
                for (int k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] += k * m_stride;
                    m_pYTensor->e(i, j, k) = X.subTensorFromTopLeft(Xc, m_filterSize).max();
                }
            }
        }
    } else if (4 == N) {
        vector<int> Xc = {0, 0, 0, 0};
        for (int i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] += i * m_stride;
            for (int j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] += j * m_stride;
                for (int k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] += k * m_stride;
                    for (int l = 0; l < m_tensorSize[3]; ++l) {
                        Xc[3] += l * m_stride;
                        m_pYTensor->e(i, j, k, l) = X.subTensorFromTopLeft(Xc, m_filterSize).max();
                    }
                }
            }
        }
    } else {
        cout << "Error: dimension>=4  does not support in MaxPooling Layer forward." << endl;
    }
}

// Y_i = max(X_i) in filterSize range
// dL/dX_i = dL/dY * 1 when Xi = max; 0 otherwise;
void MaxPoolingLayer::backward() {
    Tensor<float> &dLdy = *m_pdYTensor;
    const int N = m_filterSize.size();
    Tensor<float> &dX = *(m_prevLayer->m_pdYTensor);
    Tensor<float> &X = *(m_prevLayer->m_pYTensor);
    if (2 == N) {
        vector<int> Xc = {0, 0};
        for (int i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] += i * m_stride;
            for (int j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] += j * m_stride;
                const float maxValue = X.subTensorFromTopLeft(Xc, m_filterSize).max();
                for (int ii = 0; ii < m_filterSize[0]; ++ii)
                    for (int jj = 0; jj < m_filterSize[1]; ++jj)
                        if (maxValue == X(Xc[0] + ii, Xc[1] + jj)) dX(Xc[0] + ii, Xc[1] + jj) += dLdy(i, j);
            }
        }
    } else if (3 == N) {
        vector<int> Xc = {0, 0, 0};
        for (int i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] += i * m_stride;
            for (int j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] += j * m_stride;
                for (int k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] += k * m_stride;
                    const float maxValue = X.subTensorFromTopLeft(Xc, m_filterSize).max();
                    for (int ii = 0; ii < m_filterSize[0]; ++ii)
                        for (int jj = 0; jj < m_filterSize[1]; ++jj)
                            for (int kk = 0; kk < m_filterSize[2]; ++kk)
                                if (maxValue == X(Xc[0] + ii, Xc[1] + jj, Xc[2] + kk))
                                    dX(Xc[0] + ii, Xc[1] + jj, Xc[2] + kk) += dLdy(i, j, k);
                }
            }
        }
    } else if (4 == N) {
        vector<int> Xc = {0, 0, 0, 0};
        for (int i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] += i * m_stride;
            for (int j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] += j * m_stride;
                for (int k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] += k * m_stride;
                    for (int l = 0; l < m_tensorSize[3]; ++l) {
                        Xc[3] += l * m_stride;
                        const float maxValue = X.subTensorFromTopLeft(Xc, m_filterSize).max();
                        for (int ii = 0; ii < m_filterSize[0]; ++ii)
                            for (int jj = 0; jj < m_filterSize[1]; ++jj)
                                for (int kk = 0; kk < m_filterSize[2]; ++kk)
                                    for (int ll = 0; ll < m_filterSize[3]; ++ll)
                                        if (maxValue == X(Xc[0] + ii, Xc[1] + jj, Xc[2] + kk, Xc[3] + ll))
                                            dX(Xc[0] + ii, Xc[1] + jj, Xc[2] + kk, Xc[3] + ll) += dLdy(i, j, k, l);
                    }
                }
            }
        }
    } else {
        cout << "Error: dimension>=4  does not support in MaxPooling Layer backward." << endl;
    }
}

void MaxPoolingLayer::updateParameters(const float lr, const string &method, const int batchSize) {
    //null
}


