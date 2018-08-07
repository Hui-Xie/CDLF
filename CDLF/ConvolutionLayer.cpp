//
// Created by Hui Xie on 7/19/2018.
//

#include "ConvolutionLayer.h"
#include "statisTool.h"


ConvolutionLayer::ConvolutionLayer(const int id, const string &name, const vector<int> &filterSize,
                                   Layer *prevLayer, const int numFilters, const int stride)
        : Layer(id, name, {}) {
    if (checkFilterSize(filterSize, prevLayer)) {
        m_type = "Convolution";
        m_stride = stride;
        m_expandDy = nullptr;
        m_filterSize = filterSize;
        m_numFilters = numFilters;
        m_tensorSize = prevLayer->m_tensorSize; // this is initial, not final size

        int N = filterSize.size();
        m_OneFilterN = 1;
        for (int i = 0; i < N; ++i) {
            m_OneFilterN *= filterSize[i];
        }
        addPreviousLayer(prevLayer);
        constructFiltersAndY();
    } else {
        cout << "Error: can not construct Convolution Layer: " << name << endl;
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

    if (nullptr != m_expandDy) {
        delete m_expandDy;
        m_expandDy = nullptr;
    }

}

bool ConvolutionLayer::checkFilterSize(const vector<int> &filterSize, Layer *prevLayer) {
    int dimFilter = filterSize.size();
    int dimX = prevLayer->m_tensorSize.size();
    if (dimFilter == dimX) {
        for (int i = 0; i < dimX; ++i) {
            if (0 == filterSize[i] % 2) {
                cout << "Error: filter Size should be odd." << endl;
                return false;
            }
        }
        return true;
    } else {
        cout << "Error: dimension of filter should be consistent with the previous Layer." << endl;
        return false;
    }
}


void ConvolutionLayer::constructFiltersAndY() {
    for (int i = 0; i < m_numFilters; ++i) {
        m_pW[i] = new Tensor<float>(m_filterSize);
        m_pdW[i] = new Tensor<float>(m_filterSize);
    }

    //get refined pYTensor size
    const int dim = m_tensorSize.size();
    for (int i = 0; i < dim; ++i) {
        m_tensorSize[i] = (m_tensorSize[i] - m_filterSize[i]) / m_stride + 1;
        // ref formula: http://cs231n.github.io/convolutional-networks/
    }
    m_tensorSize.push_back(m_numFilters);

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
    const int N = m_filterSize.size();
    vector<int> Xc = m_filterSize / 2; //X central for each subTensorFromCenter at first point
    Tensor<float> &X = *m_prevLayer->m_pYTensor;
    if (2 == N) {
        for (int i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] += i * m_stride;
            for (int j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] += j * m_stride;
                Tensor<float> subX = X.subTensorFromCenter(Xc, m_filterSize);
                for (int idxF = 0; idxF < m_numFilters; ++idxF) {  //indexFilter
                    m_pYTensor->e(i, j, idxF) = subX.conv(*m_pW[idxF]);
                }
            }
        }
    } else if (3 == N) {
        for (int i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] += i * m_stride;
            for (int j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] += j * m_stride;
                for (int k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] += k * m_stride;
                    Tensor<float> subX = X.subTensorFromCenter(Xc, m_filterSize);
                    for (int idxF = 0; idxF < m_numFilters; ++idxF) {  //indexFilter
                        m_pYTensor->e(i, j, k, idxF) = subX.conv(*m_pW[idxF]);
                    }
                }
            }
        }
    } else if (4 == N) {
        for (int i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] += i * m_stride;
            for (int j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] += j * m_stride;
                for (int k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] += k * m_stride;
                    for (int l = 0; l < m_tensorSize[3]; ++l) {
                        Xc[3] += l * m_stride;
                        Tensor<float> subX = X.subTensorFromCenter(Xc, m_filterSize);
                        for (int idxF = 0; idxF < m_numFilters; ++idxF) {  //indexFilter
                            m_pYTensor->e(i, j, k, l, idxF) = subX.conv(*m_pW[idxF]);
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
    for (int idxF = 0; idxF < m_numFilters; ++idxF) {  //index of Filter
        Tensor<float> dY = m_pdYTensor->extractLowerDTensor(idxF);
        computeDW(&dY, m_pdW[idxF]);
        computeDX(&dY, m_pW[idxF]);//Note: dx need to accumulate along filters
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

void ConvolutionLayer::expandDyTensor(Tensor<float> *pdY) {
    freeExpandDy();
    const vector<int> Xdims = m_prevLayer->m_pYTensor->getDims();
    vector<int> expandDyDims = Xdims + m_filterSize;
    m_expandDy = new Tensor<float>(expandDyDims);
    m_expandDy->zeroInitialize();

    Tensor<float> &dY = *pdY;

    //copy DyTensor to expandDy, according to  m_stride
    int N = m_tensorSize.size();  //dyTensor's size
    vector<int> Xc = m_filterSize - 1; //X starting coordinate for copying dy
    if (2 == N) {
        for (int i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] += i * m_stride;
            for (int j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] += j * m_stride;
                m_expandDy->e(Xc) = dY(i, j);
            }
        }
    } else if (3 == N) {
        for (int i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] += i * m_stride;
            for (int j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] += j * m_stride;
                for (int k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] += k * m_stride;
                    m_expandDy->e(Xc) = dY(i, j, k);
                }
            }
        }
    } else if (4 == N) {
        for (int i = 0; i < m_tensorSize[0]; ++i) {
            Xc[0] += i * m_stride;
            for (int j = 0; j < m_tensorSize[1]; ++j) {
                Xc[1] += j * m_stride;
                for (int k = 0; k < m_tensorSize[2]; ++k) {
                    Xc[2] += k * m_stride;
                    for (int l = 0; l < m_tensorSize[3]; ++l) {
                        Xc[3] += l * m_stride;
                        m_expandDy->e(Xc) = dY(i, j, k, l);
                    }
                }
            }
        }
    } else {
        cout << "Error: dimension>=4  does not support in convolution expandDyTensor." << endl;
    }
}

void ConvolutionLayer::computeDW(Tensor<float> *pdY, Tensor<float> *pdW) {
    const vector<int> dWDims = pdW->getDims();
    const int N = dWDims.size();
    Tensor<float> &X = *m_prevLayer->m_pYTensor;
    const vector<int> dYDims = pdY->getDims();

    if (2 == N) {
        for (int i = 0; i < dWDims[0]; ++i) {
            for (int j = 0; j < dWDims[1]; ++j) {
                Tensor<float> Xsub = X.subTensorFromTopLeft({i * m_stride, j * m_stride}, dYDims, m_stride);
                pdW->e(i, j) += Xsub.conv(*pdY);
            }
        }
    } else if (3 == N) {
        for (int i = 0; i < dWDims[0]; ++i) {
            for (int j = 0; j < dWDims[1]; ++j) {
                for (int k = 0; k < dWDims[2]; ++k) {
                    Tensor<float> Xsub = X.subTensorFromTopLeft({i * m_stride, j * m_stride, k * m_stride}, dYDims,
                                                                m_stride);
                    pdW->e(i, j, k) += Xsub.conv(*pdY);
                }
            }
        }
    } else if (4 == N) {
        for (int i = 0; i < dWDims[0]; ++i) {
            for (int j = 0; j < dWDims[1]; ++j) {
                for (int k = 0; k < dWDims[2]; ++k) {
                    for (int l = 0; l < dWDims[3]; ++l) {
                        Tensor<float> Xsub = X.subTensorFromTopLeft(
                                {i * m_stride, j * m_stride, k * m_stride, l * m_stride}, dYDims, m_stride);
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
void ConvolutionLayer::computeDX(Tensor<float> *pdY, Tensor<float> *pW) {
    expandDyTensor(pdY);
    Tensor<float> &dX = *(m_prevLayer->m_pdYTensor);
    const vector<int> dXdims = dX.getDims();
    const int N = dXdims.size();

    if (2 == N) {
        for (int i = 0; i < dXdims[0]; ++i) {
            for (int j = 0; j < dXdims[1]; ++j) {
                Tensor<float> subExpandDy = m_expandDy->subTensorFromTopLeft({i, j}, m_filterSize, 1);
                dX(i, j) += subExpandDy.flip().conv(*pW);
            }
        }
    } else if (3 == N) {
        for (int i = 0; i < dXdims[0]; ++i) {
            for (int j = 0; j < dXdims[1]; ++j) {
                for (int k = 0; k < dXdims[2]; ++k) {
                    Tensor<float> subExpandDy = m_expandDy->subTensorFromTopLeft({i, j, k}, m_filterSize, 1);
                    dX(i, j) += subExpandDy.flip().conv(*pW);
                }
            }
        }
    } else if (4 == N) {
        for (int i = 0; i < dXdims[0]; ++i) {
            for (int j = 0; j < dXdims[1]; ++j) {
                for (int k = 0; k < dXdims[2]; ++k) {
                    for (int l = 0; l < dXdims[3]; ++l) {
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