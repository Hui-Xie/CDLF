//
// Created by Hui Xie on 11/19/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <ConvolutionBasicLayer.h>

#include "ConvolutionBasicLayer.h"
#include "statisTool.h"


ConvolutionBasicLayer::ConvolutionBasicLayer(const int id, const string &name, Layer *prevLayer,
                                             const vector<int> &filterSize, const vector<int>& stride, const int numFilters):
                                             Layer(id, name, {}){
    if (checkFilterSize(filterSize, stride, prevLayer)) {
        addPreviousLayer(prevLayer); // this must first execute
        m_type = "ConvolutionBasicLayer";
        m_stride = stride;
        m_filterSize = filterSize;
        m_numFilters = numFilters;
        m_numFeatures = m_numFilters;
        updateFeatureFilterSize();
        computeOneFiterN();

    } else {
        cout << "Error: can not construct Convolutional Layer as incorrect Filter Size." << name << endl;
    }
}

ConvolutionBasicLayer::~ConvolutionBasicLayer() {
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
bool ConvolutionBasicLayer::checkFilterSize(const vector<int> &filterSize, const vector<int>& stride, Layer *prevLayer) {
    const int dimFilter = filterSize.size();
    if (dimFilter != stride.size()){
        cout<<"Error: the dimension of filterSize and stride should be same."<<endl;
        return false;
    }

    const int dimX = prevLayer->m_tensorSize.size();
    const int s = (1 == prevLayer->m_numFeatures)? 0: 1; //indicate whether previousTensor includes feature dimension

    if ( dimFilter == dimX -s) {
        for (int i = 0; i < dimFilter; ++i) {
            if (0 == filterSize[i] % 2 && filterSize[i] != prevLayer->m_tensorSize[i+s]) {
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

void ConvolutionBasicLayer::computeOneFiterN() {
    int N = m_feature_filterSize.size();
    if (N > 0){
        m_OneFilterN = 1;
        for (int i = 0; i < N; ++i) {
            m_OneFilterN *= m_feature_filterSize[i];
        }
    }
    else{
        m_OneFilterN = 0;
    }
}

void ConvolutionBasicLayer::constructFiltersAndY() {
    m_pW = (Tensor<float> **) new void *[m_numFilters];
    m_pdW = (Tensor<float> **) new void *[m_numFilters];
    for (int i = 0; i < m_numFilters; ++i) {
        m_pW[i] = new Tensor<float>(m_feature_filterSize);
        m_pdW[i] = new Tensor<float>(m_feature_filterSize);
    }
    allocateYdYTensor();
}


void ConvolutionBasicLayer::initialize(const string &initialMethod) {
    for (int i = 0; i < m_numFilters; ++i) {
        generateGaussian(m_pW[i], 0, sqrt(1.0 / m_OneFilterN));
    }
}

void ConvolutionBasicLayer::zeroParaGradient() {
    for (int i = 0; i < m_numFilters; ++i) {
        m_pdW[i]->zeroInitialize();
    }
}


void ConvolutionBasicLayer::updateParameters(const float lr, const string &method, const int batchSize) {
    if ("sgd" == method) {
        for (int idxF = 0; idxF < m_numFilters; ++idxF) {
            *m_pW[idxF] -= *m_pdW[idxF] * (lr / batchSize);
        }
    }
}


int ConvolutionBasicLayer::getNumParameters(){
    return m_pW[0]->getLength()*m_numFilters;
}

void ConvolutionBasicLayer::save(const string &netDir) {
    FILE * pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    createDir(layerDir);

    for (int i=0; i<m_numFilters; ++i){
        filename= layerDir + "/W"+to_string(i)+".csv";
        m_pW[i]->save(filename);
    }
}

void ConvolutionBasicLayer::load(const string &netDir) {
    FILE *pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    if (!dirExist(layerDir)) {
        initialize("Xavier");
        return;
    }
    else {
        for (int i = 0; i < m_numFilters; ++i) {
            filename = layerDir + "/W" + to_string(i) + ".csv";
            m_pW[i]->load(filename);
        }
    }
}

void ConvolutionBasicLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), vector2Str(m_filterSize).c_str(), vector2Str(m_stride).c_str(), m_numFilters, 0, "{}");
}

void ConvolutionBasicLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, FilterSize=%s, Stride=%s, NumOfFilter=%d, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),  m_prevLayer->m_name.c_str(), vector2Str(m_filterSize).c_str(), vector2Str(m_stride).c_str(), m_numFilters,  vector2Str(m_tensorSize).c_str());
}

void ConvolutionBasicLayer::updateFeatureFilterSize() {
    m_feature_filterSize = m_filterSize;
    m_feature_filterSize.insert(m_feature_filterSize.begin(), m_numFeatures);
}
