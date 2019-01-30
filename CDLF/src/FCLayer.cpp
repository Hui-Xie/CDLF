//
// Created by Hui Xie on 6/5/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "FCLayer.h"
#include "statisTool.h"
#include <iostream>
#include <FCLayer.h>
#include <TensorBlas.h>


using namespace std;

// y = W*x+ b
// where y and b is m-D vector, y is output vector;
//       x is n-D input vector
//       W is m*n dimensional matrix
FCLayer::FCLayer(const int id, const string &name,  Layer *prevLayer, const int outputWidth)
: Layer(id, name,{outputWidth,1})
{
    m_type = "FCLayer";
    m_m = outputWidth;
    m_n = prevLayer->m_tensorSize[0]; //input width
    m_pW = new Tensor<float>({m_m, m_n});
    m_pBTensor = new Tensor<float>({m_m, 1});
    m_pdW = new Tensor<float>({m_m, m_n});
    m_pdBTensor = new Tensor<float>({m_m, 1});
    addPreviousLayer(prevLayer);
}

FCLayer::~FCLayer() {
    if (nullptr != m_pW) {
        delete m_pW;
        m_pW = nullptr;
    }
    if (nullptr != m_pBTensor) {
        delete m_pBTensor;
        m_pBTensor = nullptr;
    }
    if (nullptr != m_pdW) {
        delete m_pdW;
        m_pdW = nullptr;
    }
    if (nullptr != m_pdBTensor) {
        delete m_pdBTensor;
        m_pdBTensor = nullptr;
    }
}

void FCLayer::initialize(const string &initialMethod) {
    if ("Xavier" == initialMethod) {
        xavierInitialize(m_pW);
        int nRow = m_pBTensor->getDims()[0];
        const float sigmaB = 1.0 / nRow;
        generateGaussian(m_pBTensor, 0, sigmaB);
    } else {
        cout << "Error: Initialize Error in FCLayer." << endl;
    }
}

void FCLayer::zeroParaGradient() {
    if (nullptr != m_pdW) {
        m_pdW->zeroInitialize();
    }
    if (nullptr != m_pdBTensor) {
        m_pdBTensor->zeroInitialize();
    }
}

void FCLayer::forward() {
    //*m_pYTensor = (*m_pW) * (*(m_prevLayer->m_pYTensor)) + *(m_pBTensor);
    gemv(false, m_pW, m_prevLayer->m_pYTensor, m_pBTensor, m_pYTensor);
}

//   y = W*x +b
//  dL/dW = dL/dy * dy/dW = dL/dy * x'
//  dL/db = dL/dy * dy/db = dL/dy
//  dL/dx = dL/dy * dy/dx = W' * dL/dy
void FCLayer::backward(bool computeW, bool computeX) {
    Tensor<float> &dLdy = *m_pdYTensor;
    if (computeW){
       //*m_pdW += dLdy * (m_prevLayer->m_pYTensor->transpose());
        gemm(1.0, false, &dLdy, true, m_prevLayer->m_pYTensor, 1.0, m_pdW);
        //*m_pdBTensor += dLdy;
        axpy(1, &dLdy, m_pdBTensor);
    }
    if (computeX){
        //*(m_prevLayer->m_pdYTensor) += m_pW->transpose() * dLdy;
        gemv(true, m_pW, &dLdy, m_prevLayer->m_pdYTensor);
    }
}

void FCLayer::updateParameters(const float lr, const string &method, const int batchSize) {
    if ("sgd" == method) {
        //*m_pW -= (*m_pdW) * (lr / batchSize);
        matAdd(1.0, m_pW, -(lr / batchSize), m_pdW, m_pW);
        //*m_pBTensor -= (*m_pdBTensor) * (lr / batchSize);
        axpy(-(lr/batchSize), m_pdBTensor, m_pBTensor);
    }
}

void FCLayer::printWandBVector() {
    cout << "LayerType: " << m_type << "; MatrixSize " << m_m << "*" << m_n << "; W: " << endl;
    m_pW->print();
    cout << "B-transpose:" << endl;
    m_pBTensor->transpose().print();
}

void FCLayer::printdWanddBVector() {
    cout << "LayerType: " << m_type << "; MatrixSize " << m_m << "*" << m_n << "; dW: " << endl;
    m_pdW->print();
    cout << "dB-transpose:" << endl;
    m_pdBTensor->transpose().print();
}

int FCLayer::getNumParameters(){
    return m_pW->getLength() + m_pBTensor->getLength();
}

void FCLayer::save(const string &netDir) {
    FILE * pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    createDir(layerDir);

    filename= layerDir + "/W.csv";
    m_pW->save(filename);

    filename= layerDir + "/B.csv";
    m_pBTensor->save(filename);
}

void FCLayer::load(const string &netDir) {
    FILE * pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    if (!dirExist(layerDir)){
        initialize("Xavier");
        return;
    }
    else{
        filename= layerDir + "/W.csv";
        m_pW->load(filename);

        filename= layerDir + "/B.csv";
        m_pBTensor->load(filename);
    }
}

void FCLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, 0, "{}");
}

void FCLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),  m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}
