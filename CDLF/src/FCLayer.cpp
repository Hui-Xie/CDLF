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

    //m_pWLr = new Tensor<float>({m_m, m_n});
    //m_pBLr = new Tensor<float>({m_m, 1});

    m_pWM = nullptr;
    m_pBM = nullptr;
    m_pWR = nullptr;
    m_pBR = nullptr;

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

    /*
    if (nullptr != m_pWLr) {
        delete m_pWLr;
        m_pWLr = nullptr;
    }
    if (nullptr != m_pBLr) {
        delete m_pBLr;
        m_pBLr = nullptr;
    }

    */
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

/*
void FCLayer::initializeLRs(const float lr) {
    m_pWLr->uniformInitialize(lr);
    m_pBLr->uniformInitialize(lr);
}

void FCLayer::updateLRs(const float deltaLoss) {
    int N = m_pWLr->getLength();
    Tensor<float> squareGradientInv = m_pdW->hadamard(*m_pdW);
    vsInv(N, squareGradientInv.getData(), squareGradientInv.getData());
    axpy(-deltaLoss, &squareGradientInv, m_pWLr);
    //  Naive implementation
    //for (int i=0; i<N; ++i){
     //   if (0 != m_pdW->e(i)){  //todo: Important judgment,need to put into blas implementation.
     //       m_pWLr->e(i) -= deltaLossBatch/(m_pdW->e(i)*m_pdW->e(i));
     //   }
    //}


    N =  m_pBLr->getLength();
    squareGradientInv = m_pdB->hadamard(*m_pdB);
    vsInv(N, squareGradientInv.getData(), squareGradientInv.getData());
    axpy(-deltaLoss, &squareGradientInv, m_pBLr);

    // Naive imeplemenation
    //for (int i=0; i<N; ++i){
    //     if (0 != m_pdB->e(i)){
    //         m_pBLr->e(i) -= deltaLossBatch/(m_pdB->e(i)*m_pdB->e(i));
    //     }
    // }

}

*/


void FCLayer::updateParameters(Optimizer* pOptimizer) {
    if ("SGD" == pOptimizer->m_type) {
        /*  for parameter-wise learning rate
        Tensor<float> lrdw(m_pdW->getDims());
        int N = m_pW->getLength();
        vsMul(N, m_pWLr->getData(), m_pdW->getData(), lrdw.getData());
        axpy(-1.0, &lrdw, m_pW);

        Tensor<float> lrdB(m_pdB->getDims());
        N = m_pB->getLength();
        vsMul(N, m_pBLr->getData(), m_pdB->getData(), lrdB.getData());
        axpy(-1.0, &lrdB, m_pB);
        */

        SGDOptimizer* optimizer = (SGDOptimizer*) pOptimizer;
        optimizer->sgd(m_pdW, m_pW);
        optimizer->sgd(m_pdB, m_pB);

    }
    else if ("Adam" == pOptimizer->m_type){
        AdamOptimizer* optimizer = (AdamOptimizer*) pOptimizer;
        optimizer->adam(m_pWM, m_pWR, m_pdW, m_pW);
        optimizer->adam(m_pBM, m_pBR, m_pdB, m_pB);
    }
    else{
        cout<<"Error: incorrect optimizer name."<<endl;
    }
}

/*
void FCLayer::updateParameters(Optimizer* pOptimizer) {
    if ("SGD" == pOptimizer->m_type) {
        //*m_pW -= (*m_pdW) * lr;
        matAdd(1.0, m_pW, -lr, m_pdW, m_pW);
        //*m_pB -= (*m_pdB) * lr;
        axpy(-lr, m_pdB, m_pB);
    }
}
*/

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

void FCLayer::averageParaGradient(const int batchSize) {
    int N = m_pdW->getLength();
    cblas_saxpby(N, 1.0/batchSize, m_pdW->getData(), 1, 0, m_pdW->getData(), 1);
    N = m_pdB->getLength();
    cblas_saxpby(N, 1.0/batchSize, m_pdB->getData(), 1, 0, m_pdB->getData(), 1);
}

void FCLayer::allocateOptimizerMem(const string method) {
    if ("Adam" == method){
        m_pWM = new Tensor<float> (m_pW->getDims());  //1st moment
        m_pBM = new Tensor<float> (m_pB->getDims());
        m_pWR = new Tensor<float> (m_pW->getDims());  //2nd moment
        m_pBR = new Tensor<float> (m_pB->getDims());

        m_pWM->zeroInitialize();
        m_pBM->zeroInitialize();
        m_pWR->zeroInitialize();
        m_pBR->zeroInitialize();
    }
}

void FCLayer::freeOptimizerMem() {
    if (nullptr != m_pWM) {
        delete m_pWM;
        m_pWM = nullptr;
    }
    if (nullptr != m_pBM) {
        delete m_pBM;
        m_pBM = nullptr;
    }
    if (nullptr != m_pWR) {
        delete m_pWR;
        m_pWR = nullptr;
    }
    if (nullptr != m_pBR) {
        delete m_pBR;
        m_pBR = nullptr;
    }
}


