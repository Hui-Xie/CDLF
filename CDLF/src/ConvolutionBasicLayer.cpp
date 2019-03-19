//
// Created by Hui Xie on 11/19/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#include <ConvolutionBasicLayer.h>

#include "ConvolutionBasicLayer.h"
#include "statisTool.h"


ConvolutionBasicLayer::ConvolutionBasicLayer(const int id, const string &name, Layer *prevLayer,
                                             const vector<int> &filterSize, const vector<int>& stride, const int numFilters):
                                             Layer(id, name, {}){
    if (checkFilterSize(prevLayer, filterSize, stride, numFilters)) {
        m_type = "ConvolutionBasicLayer";
        m_stride = stride;
        m_filterSize = filterSize;
        m_numFilters = numFilters;
        addPreviousLayer(prevLayer);
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
    delete[] m_pW; m_pW = nullptr;
    delete[] m_pdW; m_pdW = nullptr;

    if (nullptr != m_pB){
        delete m_pB;
        m_pB = nullptr;
    }
    if (nullptr != m_pdB){
        delete m_pdB;
        m_pdB = nullptr;
    }

    /*
    // for parameter-wise learning rate
    for (int i = 0; i < m_numFilters; ++i) {
        if (nullptr != m_pWLr[i]) {
            delete m_pWLr[i];
            m_pWLr[i] = nullptr;
        }
    }
    delete[] m_pWLr; m_pWLr = nullptr;

    if (nullptr != m_pBLr){
        delete m_pBLr;
        m_pBLr = nullptr;
    }
*/
}

// the filterSize in each dimension should be odd,
// or if it is even, it must be same size of corresponding dimension of tensorSize of input X

bool ConvolutionBasicLayer::checkFilterSize(Layer *prevLayer, const vector<int> &filterSize, const vector<int> &stride,
                                            const int numFilters) {
    const int dimFilter = filterSize.size();
    if (dimFilter != stride.size()) {
        cout << "Error: the dimension of filterSize and stride should be same." << endl;
        return false;
    }

    if (!isElementBiggerThan0(stride)){
        cout<<"Error: the stride should be greater than 0. "<<endl;
        return false;
    }

    const int dimX = prevLayer->m_tensorSize.size();
    if (dimX == dimFilter) {
        m_numInputFeatures = 1;
    } else if (dimX - 1 == dimFilter) {
        m_numInputFeatures = prevLayer->m_tensorSize[0];
    } else {
        cout<< "Error: the dimensionon of filterSize should be equal with or 1 less than of the dimension of previous layer tensorSize."
            << endl;
        return false;
    }

    const int s = (1 == m_numInputFeatures) ? 0 : 1; //indicate whether previousTensor includes feature dimension


    for (int i = 0; i < dimFilter; ++i) {
        if (0 == filterSize[i] % 2 && filterSize[i] != prevLayer->m_tensorSize[i + s]) {
            cout << "Error: the filterSize in each dimension should be odd, "
                    "or if it is even, it must be same size of corresponding dimension of tensorSize of input X."
                 << endl;
            return false;
        }
    }
    return true;

}

void ConvolutionBasicLayer::updateFeatureFilterSize() {
    m_feature_filterSize = m_filterSize;
    m_feature_filterSize.insert(m_feature_filterSize.begin(), m_numInputFeatures);
    m_feature_stride = m_stride;
    m_feature_stride.insert(m_feature_stride.begin(), 1);
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

        if (1 != m_numInputFeatures){
            m_pW[i] = new Tensor<float>(m_feature_filterSize);
            m_pdW[i] = new Tensor<float>(m_feature_filterSize);
        }
        else{
            m_pW[i] = new Tensor<float>(m_filterSize);
            m_pdW[i] = new Tensor<float>(m_filterSize);
        }

    }

    m_pB = new Tensor<float>({m_numFilters,1});
    m_pdB = new Tensor<float>({m_numFilters,1});

    allocateYdYTensor();

    /*
    //for parameter-wise learning rates
    m_pBLr = new Tensor<float>({m_numFilters,1});
    m_pWLr = (Tensor<float> **) new void *[m_numFilters];
    for (int i = 0; i < m_numFilters; ++i) {
        if (1 != m_numInputFeatures){
            m_pWLr[i] = new Tensor<float>(m_feature_filterSize);
        }
        else{
            m_pWLr[i] = new Tensor<float>(m_filterSize);
        }
    }
    */
}


void ConvolutionBasicLayer::initialize(const string &initialMethod) {
    for (int i = 0; i < m_numFilters; ++i) {
        generateGaussian(m_pW[i], 0, sqrt(1.0 / m_OneFilterN));
    }
    generateGaussian(m_pB, 0, 1.0 / m_numFilters);
}

void ConvolutionBasicLayer::zeroParaGradient() {
    for (int i = 0; i < m_numFilters; ++i) {
        m_pdW[i]->zeroInitialize();
    }
    m_pdB->zeroInitialize();
}


void ConvolutionBasicLayer::computeDb(const Tensor<float> *pdY, const int filterIndex) {
    m_pdB->e(filterIndex) += pdY->sum(); // + is for batch processing
}

/*
void ConvolutionBasicLayer::initializeLRs(const float lr) {
    for (int i = 0; i < m_numFilters; ++i) {
        m_pWLr[i]->uniformInitialize(lr);
    }
    m_pBLr->uniformInitialize(lr);
}

void ConvolutionBasicLayer::updateLRs(const float deltaLoss) {

}

void ConvolutionBasicLayer::updateParameters(const string& method, Optimizer* pOptimizer) {

}
*/

void ConvolutionBasicLayer::updateParameters(const string& method, Optimizer* pOptimizer) {
    if ("sgd" == method) {
        SGDOptimizer* sgdOptimizer = (SGDOptimizer*) pOptimizer;
        for (int idxF = 0; idxF < m_numFilters; ++idxF) {
            sgdOptimizer->sgd(m_pdW[idxF], m_pW[idxF]);
        }
        sgdOptimizer->sgd(m_pdB, m_pB);
    }
    else if ("Adam" == method){

    }
    else {

    }
}


int ConvolutionBasicLayer::getNumParameters(){
    return m_pW[0]->getLength()*m_numFilters + m_pB->getLength();
}

void ConvolutionBasicLayer::save(const string &netDir) {
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    createDir(layerDir);

    for (int i=0; i<m_numFilters; ++i){
        filename= layerDir + "/W"+to_string(i)+".csv";
        m_pW[i]->save(filename);
    }

    filename= layerDir + "/B.csv";
    m_pB->save(filename);
}

void ConvolutionBasicLayer::load(const string &netDir) {
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    if (!dirExist(layerDir)) {
        initialize("Xavier");
        return;
    }
    else {
        for (int i = 0; i < m_numFilters; ++i) {
            filename = layerDir + "/W" + to_string(i) + ".csv";
            if (! m_pW[i]->load(filename)){
                generateGaussian(m_pW[i], 0, sqrt(1.0 / m_OneFilterN));
            }
        }

        filename = layerDir + "/B.csv";
        if (! m_pB->load(filename)){
            generateGaussian(m_pB, 0, sqrt(1.0 / m_numFilters));
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

void ConvolutionBasicLayer::averageParaGradient(const int batchSize) {

}



/*  for bias cudnn test
void ConvolutionBasicLayer::beforeGPUCheckdBAnddY() {
    cout<<m_name<<endl;
    cout<<"dB : "<<endl;
     m_pdB->print();
     Tensor<float> dYSum({m_numFilters,1});
     const int N= length(m_tensorSize)/m_numFilters;
     for (int i=0; i<m_numFilters; ++i){
         dYSum.e(i) =0;
         for(int j=0; j<N; ++j){
             dYSum.e(i) += m_pdYTensor->e(i*N+j);
         }
     }
     cout<<"dYSum: "<<endl;
     dYSum.print();
     cout<<"updated db should be: "<<endl;
    (*m_pdB+ dYSum).print();

}

void ConvolutionBasicLayer::afterGPUCheckdB() {
    cout<<"After GPU, dB : "<<endl;
    m_pdB->print();
    cout<<"================="<<endl;
}

*/


