//
// Created by Hui Xie on 11/29/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <MeanSquareLossLayer.h>

MeanSquareLossLayer::MeanSquareLossLayer(const int id, const string &name, Layer *prevLayer, float lambda) : LossLayer(id, name, prevLayer) {
    m_type = "MeanSquareLossLayer";
    m_lambda = lambda;
}

MeanSquareLossLayer::~MeanSquareLossLayer() {
   //null;
}

void MeanSquareLossLayer::printGroundTruth() {
    if (nullptr != m_pGroundTruth){
        m_pGroundTruth->print();
    }
}

//  L= lambda*(0.5/N)*\sum (x_i- g_i)^2
float MeanSquareLossLayer::lossCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    const int N = X.getLength();
    float loss = 0.0;
    for (int i=0; i<N; ++i){
        loss += pow(X.e(i) - m_pGroundTruth->e(i), 2);
    }
    m_loss = loss*0.5*m_lambda/N;
    return m_loss;
 }

//dL/dx = (x_i-g_i)*lambda/N
void MeanSquareLossLayer::gradientCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    Tensor<float> & dX = *(m_prevLayer->m_pdYTensor);
    const int N = X.getLength();
    float lambda_N = m_lambda/N;
    for (int i=0; i<N; ++i){
       dX[i] += (X.e(i) - m_pGroundTruth->e(i))*lambda_N;
    }
}

void MeanSquareLossLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, previousLayerIDs, outputTensorSize, filterSize, numFilter, FilterStride, startPosition, \r\n";
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %f, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, m_lambda, "{}");
}

void MeanSquareLossLayer::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s, Type=%s, id=%d, PrevLayer=%s; Lambda=%f; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), m_lambda);
}

float MeanSquareLossLayer::diceCoefficient(const float threshold) {
    const Tensor<float>* pPredict = m_prevLayer->m_pYTensor;
    const Tensor<float>* pGT =  m_pGroundTruth;
    const int N = pPredict->getLength();
    int nSuccess = 0;
    for (int i=0; i< N; ++i)
    {
        if (pPredict->e(i) > threshold && pGT->e(i) > threshold){
            ++nSuccess;
        }
    }
    return nSuccess*1.0/N;
}
