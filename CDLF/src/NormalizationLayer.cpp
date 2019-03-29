//
// Created by Hui Xie on 6/12/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#include "NormalizationLayer.h"
#include "statisTool.h"
#include <math.h>       /* sqrt */
#include <NormalizationLayer.h>
#include "ConvolutionBasicLayer.h"


NormalizationLayer::NormalizationLayer(const int id, const string& name,Layer* prevLayer, const vector<int>& tensorSize):Layer(id,name, tensorSize){
    m_type = "NormalizationLayer";
    addPreviousLayer(prevLayer);
    m_epsilon = 1e-6;
    m_pSigma = nullptr;
    m_existFeautureDim = false;

    if (length(m_tensorSize) != length(m_prevLayer->m_tensorSize)){
        cout<<"Error: The output TensorSize does not equal with the one of the previous layer in Normalization construction at layID = "<<id<<endl;
    }
    setFeatureDim();
    allocateSigmas();
}
NormalizationLayer::~NormalizationLayer(){
   freeSigmas();
}

void NormalizationLayer::initialize(const string& initialMethod){
     //null
}

void NormalizationLayer::zeroParaGradient(){
    //null
}

// Y = (X-mu)/sigma
// sigma = sigma + m_epsilon,   avoid sigma == 0
// dL/dX = dL/dY * dY/dX = dL/dY * (1/sigma)
void NormalizationLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    if (!m_existFeautureDim){
        float mean = X.average();
        m_pSigma[0] = sqrt(X.variance());
        m_pSigma[0] = (0.0 == m_pSigma[0])? m_epsilon: m_pSigma[0];
        const int N = Y.getLength();
        for (int i=0; i<N; ++i){
            Y.e(i) = (X.e(i) -mean)/m_pSigma[0];
        }
    }
    else{
        const int numFeatures = m_tensorSize[0];
        const int N = length(m_tensorSize)/numFeatures;
        for (int j=0; j< numFeatures; ++j){
            Tensor<float>* pTensor = nullptr;
            X.extractLowerDTensor(j, pTensor);

            float mean = pTensor->average();
            m_pSigma[j] = sqrt(pTensor->variance());
            delete pTensor;

            m_pSigma[j] = (0.0 == m_pSigma[j])? m_epsilon: m_pSigma[j];
            for (int i=j*N; i<(j+1)*N; ++i){
                Y.e(i) = (X.e(i) -mean)/m_pSigma[j];
            }
        }
    }

    // below sentense will change Y's dims.discard
    //Y = (X-mean)/m_sigma;

}
void NormalizationLayer::backward(bool computeW, bool computeX){
    if(computeX){
        Tensor<float>& dY = *m_pdYTensor;
        Tensor<float>& dX = *(m_prevLayer->m_pdYTensor);
        if (!m_existFeautureDim){
            const int N = dX.getLength();
            for (int i=0; i<N; ++i){
                dX.e(i) += dY.e(i)/m_pSigma[0];
            }
        }
        else{
            const int numFeatures = m_tensorSize[0];
            const int N = length(m_tensorSize)/numFeatures;
            for (int j=0; j< numFeatures; ++j){
                for (int i=j*N; i<(j+1)*N; ++i){
                    dX.e(i) += dY.e(i)/m_pSigma[j];
                }
            }

        }
    }
}


int  NormalizationLayer::getNumParameters(){
    return 0;
}

void NormalizationLayer::save(const string &netDir) {
  //null;
}

void NormalizationLayer::load(const string &netDir) {
  //null;
}

void NormalizationLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", "{}", 0, 0, "{}");
}

void NormalizationLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),  m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}

void NormalizationLayer::initializeLRs(const float lr) {

}

void NormalizationLayer::updateLRs(const float deltaLoss) {

}

void NormalizationLayer::updateParameters(Optimizer* pOptimizer) {

}

void NormalizationLayer::averageParaGradient(const int batchSize) {

}

void NormalizationLayer::setFeatureDim() {
    if (nullptr != m_prevLayer){
        if (   (nullptr != m_prevLayer->m_prevLayer)
            && ("TransposedConvolutionLayer" == m_prevLayer->m_prevLayer->m_type || "ConvolutionLayer" == m_prevLayer->m_prevLayer->m_type )
            && (1 != ((ConvolutionBasicLayer*) m_prevLayer->m_prevLayer)->m_numFilters))
        {
            m_existFeautureDim = true;
        }
        else if (("TransposedConvolutionLayer" == m_prevLayer->m_type || "ConvolutionLayer" == m_prevLayer->m_type )
                 && (1 != ((ConvolutionBasicLayer*) m_prevLayer)->m_numFilters))
        {
            m_existFeautureDim = true;
        }
        else{
            m_existFeautureDim = false;
        }
    }
    else{
        m_existFeautureDim = false;
    }

}

void NormalizationLayer::allocateSigmas() {
   if (m_existFeautureDim){
       const int N = m_tensorSize[0];
       m_pSigma = new float[N];
   }
   else{
       m_pSigma = new float[1];
   }
}

void NormalizationLayer::freeSigmas() {
    if (nullptr != m_pSigma){
        delete[] m_pSigma;
        m_pSigma = nullptr;
    }
}
