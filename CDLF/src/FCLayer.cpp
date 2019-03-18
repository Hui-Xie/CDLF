//
// Created by Hui Xie on 6/5/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

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
    m_pB = new Tensor<float>({m_m, 1});
    m_pdW = new Tensor<float>({m_m, m_n});
    m_pdB = new Tensor<float>({m_m, 1});

    m_pWLr = new Tensor<float>({m_m, m_n});
    m_pBLr = new Tensor<float>({m_m, 1});

    addPreviousLayer(prevLayer);
}

FCLayer::~FCLayer() {
    if (nullptr != m_pW) {
        delete m_pW;
        m_pW = nullptr;
    }
    if (nullptr != m_pB) {
        delete m_pB;
        m_pB = nullptr;
    }
    if (nullptr != m_pdW) {
        delete m_pdW;
        m_pdW = nullptr;
    }
    if (nullptr != m_pdB) {
        delete m_pdB;
        m_pdB = nullptr;
    }

    if (nullptr != m_pWLr) {
        delete m_pWLr;
        m_pWLr = nullptr;
    }
    if (nullptr != m_pBLr) {
        delete m_pBLr;
        m_pBLr = nullptr;
    }
}

void FCLayer::initialize(const string &initialMethod) {
    if ("Xavier" == initialMethod) {
        xavierInitialize(m_pW);
        int nRow = m_pB->getDims()[0];
        const float sigmaB = 1.0 / nRow;
        generateGaussian(m_pB, 0, sigmaB);
    } else {
        cout << "Error: Initialize Error in FCLayer." << endl;
    }
}

void FCLayer::zeroParaGradient() {
    if (nullptr != m_pdW) {
        m_pdW->zeroInitialize();
    }
    if (nullptr != m_pdB) {
        m_pdB->zeroInitialize();
    }
}

void FCLayer::forward() {
    //*m_pYTensor = (*m_pW) * (*(m_prevLayer->m_pYTensor)) + *(m_pB);
    gemv(false, m_pW, m_prevLayer->m_pYTensor, m_pB, m_pYTensor);
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
        //*m_pdB += dLdy;
        axpy(1, &dLdy, m_pdB);
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
        //*m_pB -= (*m_pdB) * (lr / batchSize);
        axpy(-(lr/batchSize), m_pdB, m_pB);
    }
}

void FCLayer::updateParameters(const string &method, const int batchSize) {
    if ("sgd" == method) {
        const float batchSizeInv = 1.0/batchSize;
        Tensor<float> gradientW(m_pdW->getDims());
        gradientW.zeroInitialize();
        axpy(batchSizeInv, m_pdW, &gradientW);
        int N = m_pW->getLength();
        vsMul(N, m_pWLr->getData(), gradientW.getData(), gradientW.getData());
        axpy(-1.0, &gradientW, m_pW);

        Tensor<float> gradientB(m_pdB->getDims());
        gradientB.zeroInitialize();
        axpy(batchSizeInv, m_pdB, &gradientB);
        N = m_pB->getLength();
        vsMul(N, m_pBLr->getData(), gradientB.getData(), gradientB.getData());
        axpy(-1.0, &gradientB, m_pB);
    }
}


void FCLayer::printWandBVector() {
    cout << "LayerType: " << m_type << "; MatrixSize " << m_m << "*" << m_n << "; W: " << endl;
    m_pW->print();
    cout << "B-transpose:" << endl;
    m_pB->transpose().print();
}

void FCLayer::printdWanddBVector() {
    cout << "LayerType: " << m_type << "; MatrixSize " << m_m << "*" << m_n << "; dW: " << endl;
    m_pdW->print();
    cout << "dB-transpose:" << endl;
    m_pdB->transpose().print();
}

int FCLayer::getNumParameters(){
    return m_pW->getLength() + m_pB->getLength();
}

void FCLayer::save(const string &netDir) {
    FILE * pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    createDir(layerDir);

    filename= layerDir + "/W.csv";
    m_pW->save(filename);

    filename= layerDir + "/B.csv";
    m_pB->save(filename);
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
        if (!m_pW->load(filename)){
            xavierInitialize(m_pW);
        }

        filename= layerDir + "/B.csv";
        if (!m_pB->load(filename)){
            int nRow = m_pB->getDims()[0];
            const float sigmaB = 1.0 / nRow;
            generateGaussian(m_pB, 0, sigmaB);
        }
    }
}

void FCLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", "{}", 0, 0, "{}");
}

void FCLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),  m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}

void FCLayer::initializeLRs(const float lr) {
    m_pWLr->uniformInitialize(lr);
    m_pBLr->uniformInitialize(lr);
}

void FCLayer::updateLRs(const float deltaLoss, const int batchSize) {
   float deltaLossBatch = deltaLoss* batchSize* batchSize;

   int N = m_pWLr->getLength();
   Tensor<float> squareGradientInv = m_pdW->hadamard(*m_pdW);
   vsInv(N, squareGradientInv.getData(), squareGradientInv.getData());
   axpy(-deltaLossBatch, &squareGradientInv, m_pWLr);
   /*  Naive implementation
   for (int i=0; i<N; ++i){
       if (0 != m_pdW->e(i)){
           m_pWLr->e(i) -= deltaLossBatch/(m_pdW->e(i)*m_pdW->e(i));
       }
   }
   */

   N =  m_pBLr->getLength();
   squareGradientInv = m_pdB->hadamard(*m_pdB);
   vsInv(N, squareGradientInv.getData(), squareGradientInv.getData());
   axpy(-deltaLossBatch, &squareGradientInv, m_pBLr);

   /* Naive imeplemenation
   for (int i=0; i<N; ++i){
        if (0 != m_pdB->e(i)){
            m_pBLr->e(i) -= deltaLossBatch/(m_pdB->e(i)*m_pdB->e(i));
        }
    }
   */
}
