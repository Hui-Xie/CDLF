//
// Created by Hui Xie on 6/5/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "FCLayer.h"
#include "statisTool.h"
#include <iostream>
#include <FCLayer.h>


using namespace std;

// y = W*x+ b
// where y and b is m-D vector, y is output vector;
//       x is n-D input vector
//       W is m*n dimensional matrix
FCLayer::FCLayer(const int id, const string &name,  Layer *prevLayer, const long outputWidth)
: Layer(id, name,{outputWidth,1})
{
    m_type = "FullyConnected";
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
        long nRow = m_pBTensor->getDims()[0];
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
    *m_pYTensor = (*m_pW) * (*(m_prevLayer->m_pYTensor)) + *(m_pBTensor);
}

//   y = W*x +b
//  dL/dW = dL/dy * dy/dW = dL/dy * x'
//  dL/db = dL/dy * dy/db = dL/dy
//  dL/dx = dL/dy * dy/dx = W' * dL/dy
void FCLayer::backward(bool computeW) {
    Tensor<float> &dLdy = *m_pdYTensor;
    if (computeW){
        *m_pdW += dLdy * (m_prevLayer->m_pYTensor->transpose());
        *m_pdBTensor += dLdy;
    }
    *(m_prevLayer->m_pdYTensor) += m_pW->transpose() * dLdy;
}

void FCLayer::updateParameters(const float lr, const string &method, const int batchSize) {
    if ("sgd" == method) {
        *m_pW -= (*m_pdW) * (lr / batchSize);
        *m_pBTensor -= (*m_pdBTensor) * (lr / batchSize);
    }
}

void FCLayer::printWandBVector() {
    cout << "LayerType: " << m_type << "; MatrixSize " << m_m << "*" << m_n << "; W: " << endl;
    m_pW->printElements();
    cout << "B-transpose:" << endl;
    m_pBTensor->transpose().printElements();
}

void FCLayer::printdWanddBVector() {
    cout << "LayerType: " << m_type << "; MatrixSize " << m_m << "*" << m_n << "; dW: " << endl;
    m_pdW->printElements();
    cout << "dB-transpose:" << endl;
    m_pdBTensor->transpose().printElements();
}

long FCLayer::getNumParameters(){
    return m_pW->getLength() + m_pBTensor->getLength();
}

void FCLayer::save(const string &netDir) {

}

void FCLayer::load(const string &netDir) {

}
