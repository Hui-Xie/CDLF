//
// Created by Hui Xie on 7/19/2018.
//

#include "ConvolutionLayer.h"
#include "statisTool.h"

ConvolutionLayer::ConvolutionLayer(const int id, const string &name, const vector<long> &filterSize,
                                   Layer *prevLayer, const int numFilters, const int stride)
        : Layer(id, name, {}) {
    if (checkFilterSize(filterSize, prevLayer)) {
        m_type = "Convolution";
        m_stride = stride;
        m_expandDy = nullptr;
        m_filterSize = filterSize;
        m_numFilters = numFilters;
        addPreviousLayer(prevLayer);
        computeOneFiterN();
        updateTensorSize();
        constructFiltersAndY();
    } else {
        cout << "Error: can not construct Convolution Layer as incorrect Filter Size." << name << endl;
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

    if (nullptr != m_expandDy) {
        delete m_expandDy;
        m_expandDy = nullptr;
    }
}

bool ConvolutionLayer::checkFilterSize(const vector<long> &filterSize, Layer *prevLayer) {
    int dimFilter = filterSize.size();
    int dimX = prevLayer->m_tensorSize.size();
    if (dimFilter == dimX) {
        for (int i = 0; i < dimX; ++i) {
            if (0 == filterSize[i] % 2) {
                cout << "Error: filter Size should be odd." << endl;  // this is a better design
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
    m_OneFilterN = 1;
    for (int i = 0; i < N; ++i) {
        m_OneFilterN *= m_filterSize[i];
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
    m_pW = (Tensor<float>**) new void* [m_numFilters];
    m_pdW = (Tensor<float>**) new void* [m_numFilters];
    for (int i = 0; i < m_numFilters; ++i) {
        m_pW[i] = new Tensor<float>(m_filterSize);
        m_pdW[i] = new Tensor<float>(m_filterSize);
    }

    if (0 != m_tensorSize.size()) {
        m_pYTensor = new Tensor<float>(m_tensorSize);
        m_pdYTensor = new Tensor<float>(m_tensorSize);
    } else {
        m_pYTensor = nullptr;
        m_pdYTensor = nullptr;
    }
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
    const int Df = ( 1== m_numFilters)? 0:1; //Feature dimension

    vector<long> f = nonZeroIndex(m_prevLayer->m_tensorSize - m_filterSize);

    vector<long> Xs = m_filterSize / 2; //X central for each subTensorFromCenter at first point
    vector<long> Xc = Xs;
    Tensor<float> &X = *m_prevLayer->m_pYTensor;
    if (2 == Nt -Df) {
        for (long i = 0; i < m_tensorSize[Df+0]; ++i) {
            Xc[f[0]] = Xs[f[0]] + i * m_stride;
            for (long j = 0; j < m_tensorSize[Df+1]; ++j) {
                Xc[f[1]] = Xs[f[1]] + j * m_stride;
                Tensor<float> subX = X.subTensorFromCenter(Xc, m_filterSize);
                if (1 != m_numFilters) {
                    for (int idxF = 0; idxF < m_numFilters; ++idxF) {  //indexFilter
                        m_pYTensor->e(idxF, i, j) = subX.conv(*m_pW[idxF]);
                    }
                } else {
                    m_pYTensor->e(i, j) = subX.conv(*m_pW[0]);
                }

            }
        }
    } else if (3 == Nt-Df) {
        for (long i = 0; i < m_tensorSize[Df+0]; ++i) {
            Xc[f[0]] = Xs[f[0]] + i * m_stride;
            for (long j = 0; j < m_tensorSize[Df+1]; ++j) {
                Xc[f[1]] = Xs[f[1]] + j * m_stride;
                for (long k = 0; k < m_tensorSize[Df+2]; ++k) {
                    Xc[f[2]] = Xs[f[2]] + k * m_stride;
                    Tensor<float> subX = X.subTensorFromCenter(Xc, m_filterSize);
                    if (1 != m_numFilters) {
                        for (int idxF = 0; idxF < m_numFilters; ++idxF) {  //indexFilter
                            m_pYTensor->e(idxF, i, j, k) = subX.conv(*m_pW[idxF]);
                        }
                    } else {
                        m_pYTensor->e(i, j, k) = subX.conv(*m_pW[0]);
                    }

                }
            }
        }
    } else if (4 == Nt-Df) {
        for (long i = 0; i < m_tensorSize[Df+0]; ++i) {
            Xc[f[0]] = Xs[f[0]] + i * m_stride;
            for (long j = 0; j < m_tensorSize[Df+1]; ++j) {
                Xc[f[1]] = Xs[f[1]] + j * m_stride;
                for (long k = 0; k < m_tensorSize[Df+2]; ++k) {
                    Xc[f[2]] = Xs[f[2]] + k * m_stride;
                    for (long l = 0; l < m_tensorSize[Df+3]; ++l) {
                        Xc[f[3]] = Xs[f[3]] + l * m_stride;
                        Tensor<float> subX = X.subTensorFromCenter(Xc, m_filterSize);
                        if (1 != m_numFilters) {
                            for (int idxF = 0; idxF < m_numFilters; ++idxF) {  //indexFilter
                                m_pYTensor->e(idxF, i, j, k, l) = subX.conv(*m_pW[idxF]);
                            }
                        } else {
                            m_pYTensor->e(i, j, k, l) = subX.conv(*m_pW[0]);
                        }

                    }
                }
            }
        }
    } else {
        cout << "Error: dimension>=4  does not support in convolution forward." << endl;
    }
}

// Y =W*X
// dL/dW = dL/dY * dY/dW;
// dL/dX = dL/dY * dY/dX;
// algorithm ref: https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
void ConvolutionLayer::backward() {
    // dX needs to consider the accumulation of different filters
    if (1 != m_numFilters) {
        for (int idxF = 0; idxF < m_numFilters; ++idxF) {  //index of Filter
            Tensor<float> dY = m_pdYTensor->extractLowerDTensor(idxF);
            computeDW(&dY, m_pdW[idxF]);
            computeDX(&dY, m_pW[idxF]);//Note: dx need to accumulate along filters
        }
    } else {
        computeDW(m_pdYTensor, m_pdW[0]);
        computeDX(m_pdYTensor, m_pW[0]);
    }
}

void ConvolutionLayer::updateParameters(const float lr, const string &method, const int batchSize) {
    if ("sgd" == method) {
        for (int idxF = 0; idxF < m_numFilters; ++idxF) {
            *m_pW[idxF] -= *m_pdW[idxF] * (lr / batchSize);
        }
    }
}

void ConvolutionLayer::freeExpandDy() {
    if (nullptr != m_expandDy) {
        delete m_expandDy;
        m_expandDy = nullptr;
    }
}

void ConvolutionLayer::expandDyTensor(const Tensor<float> *pdY) {
    freeExpandDy();
    const vector<long> Xdims = m_prevLayer->m_pYTensor->getDims();
    vector<long> expandDyDims = Xdims + m_filterSize;
    m_expandDy = new Tensor<float>(expandDyDims);
    m_expandDy->zeroInitialize();

    const Tensor<float> &dY = *pdY;

    //copy DyTensor to expandDy, according to  m_stride
    int N = m_tensorSize.size();  //dyTensor's size
    vector<long> Xs = m_filterSize - 1; //X starting coordinate for copying dy
    vector<long> Xc = m_filterSize * 0;
    if (2 == N) {
        for (long i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] = Xs[0] + i * m_stride;
            for (long j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] = Xs[1] + j * m_stride;
                m_expandDy->e(Xc) = dY(i, j);
            }
        }
    } else if (3 == N) {
        for (long i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] = Xs[0] + i * m_stride;
            for (long j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] = Xs[1] + j * m_stride;
                for (long k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] = Xs[2] + k * m_stride;
                    m_expandDy->e(Xc) = dY(i, j, k);
                }
            }
        }
    } else if (4 == N) {
        for (long i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] = Xs[0] + i * m_stride;
            for (long j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] = Xs[1] + j * m_stride;
                for (long k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] = Xs[2] + k * m_stride;
                    for (long l = 0; l < m_tensorSize[3]; ++l) {
                        Xc[3] = Xs[3] + l * m_stride;
                        m_expandDy->e(Xc) = dY(i, j, k, l);
                    }
                }
            }
        }
    } else {
        cout << "Error: dimension>=4  does not support in convolution expandDyTensor." << endl;
    }
}

void ConvolutionLayer::computeDW(const Tensor<float> *pdY, Tensor<float> *pdW) {
    const vector<long> dWDims = pdW->getDims();
    const int Nf = dWDims.size();
    Tensor<float> &X = *m_prevLayer->m_pYTensor;
    const vector<long> dYDims = pdY->getDims();

    vector<long> f = nonZeroIndex(m_prevLayer->m_tensorSize - m_filterSize);
    vector<long> dYDimsEx(Nf,1); //Nf longs with value of 1
    for (int i=0; i< dYDims.size(); ++i){
        dYDimsEx[f[i]] = dYDims[i];
    }

    if (2 == Nf) {
        for (int i = 0; i < dWDims[0]; ++i) {
            for (int j = 0; j < dWDims[1]; ++j) {
                Tensor<float> Xsub = X.subTensorFromTopLeft({i * m_stride, j * m_stride}, dYDimsEx, m_stride);
                pdW->e(i, j) += Xsub.conv(*pdY); // + is for batch processing
            }
        }
    } else if (3 == Nf) {
        for (int i = 0; i < dWDims[0]; ++i) {
            for (int j = 0; j < dWDims[1]; ++j) {
                for (int k = 0; k < dWDims[2]; ++k) {
                    Tensor<float> Xsub = X.subTensorFromTopLeft({i * m_stride, j * m_stride, k * m_stride}, dYDimsEx,
                                                                m_stride);
                    pdW->e(i, j, k) += Xsub.conv(*pdY);
                }
            }
        }
    } else if (4 == Nf) {
        for (int i = 0; i < dWDims[0]; ++i) {
            for (int j = 0; j < dWDims[1]; ++j) {
                for (int k = 0; k < dWDims[2]; ++k) {
                    for (int l = 0; l < dWDims[3]; ++l) {
                        Tensor<float> Xsub = X.subTensorFromTopLeft(
                                {i * m_stride, j * m_stride, k * m_stride, l * m_stride}, dYDimsEx, m_stride);
                        pdW->e(i, j, k, l) += Xsub.conv(*pdY);
                    }
                }
            }
        }
    } else {
        cout << "Error: Dimension >=5 does not support in ConvolutionLayer::computeDW" << endl;
    }
}

//Note: dx need to accumulate along filters
void ConvolutionLayer::computeDX(const Tensor<float> *pdY, const Tensor<float> *pW) {
    expandDyTensor(pdY);
    Tensor<float> &dX = *(m_prevLayer->m_pdYTensor);
    const vector<long> dXdims = dX.getDims();
    const int N = dXdims.size();

    if (2 == N) {
        for (long i = 0; i < dXdims[0]; ++i) {
            for (long j = 0; j < dXdims[1]; ++j) {
                Tensor<float> subExpandDy = m_expandDy->subTensorFromTopLeft({i, j}, m_filterSize, 1);
                dX(i, j) += subExpandDy.flip().conv(*pW);
            }
        }
    } else if (3 == N) {
        for (long i = 0; i < dXdims[0]; ++i) {
            for (long j = 0; j < dXdims[1]; ++j) {
                for (long k = 0; k < dXdims[2]; ++k) {
                    Tensor<float> subExpandDy = m_expandDy->subTensorFromTopLeft({i, j, k}, m_filterSize, 1);
                    dX(i, j) += subExpandDy.flip().conv(*pW);
                }
            }
        }
    } else if (4 == N) {
        for (long i = 0; i < dXdims[0]; ++i) {
            for (long j = 0; j < dXdims[1]; ++j) {
                for (long k = 0; k < dXdims[2]; ++k) {
                    for (long l = 0; l < dXdims[3]; ++l) {
                        Tensor<float> subExpandDy = m_expandDy->subTensorFromTopLeft({i, j, k, l}, m_filterSize, 1);
                        dX(i, j) += subExpandDy.flip().conv(*pW);
                    }
                }
            }
        }
    } else {
        cout << "Error: Dimension >=5 does not support in ConvolutionLayer::computeDX." << endl;
    }
    freeExpandDy();
}
