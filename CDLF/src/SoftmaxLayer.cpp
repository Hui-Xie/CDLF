//
// Created by Hui Xie on 7/28/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#include <math.h>       /* exp */
#include <SoftmaxLayer.h>

#ifdef Use_GPU
     #include <CudnnSoftmax.h>
#endif


SoftmaxLayer::SoftmaxLayer(const int id, const string& name,Layer* prevLayer):Layer(id,name, prevLayer->m_tensorSize) {
    m_type = "SoftmaxLayer";
    addPreviousLayer(prevLayer);
}

SoftmaxLayer::~SoftmaxLayer(){

}

void SoftmaxLayer::initialize(const string& initialMethod){
    //null
}

void SoftmaxLayer::zeroParaGradient(){
    //null
}

void SoftmaxLayer::forward(){
#ifdef Use_GPU
    CudnnSoftmax cudnnSoftmax(this);
    cudnnSoftmax.forward();
#else

    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    const int nSoftmax = m_pYTensor->getDims()[0];// a vector's dimension to execute softmax
    const int N = X.getLength()/nSoftmax;  // the number of element vectors needing softmax
    for (int j=0; j<N; ++j){
        float sumExpX = 0;
        for (int i=0; i< nSoftmax; ++i){
            sumExpX += exp(X(i*N+j));
        }

        for (int i=0; i< nSoftmax; ++i){
            Y(i*N+j) = exp(X(i*N+j))/sumExpX;
        }
    }
#endif
}

void SoftmaxLayer::backward(bool computeW, bool computeX){
    if (!computeX) return;

#ifdef Use_GPU
    CudnnSoftmax cudnnSoftmax(this);
    cudnnSoftmax.backward(computeW, computeX);

#else

    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    const int nSoftmax = m_pdYTensor->getDims()[0];// a vector's dimension to execute softmax
    const int N = X.getLength()/nSoftmax;  // the number of element vectors needing softmax
    for (int j=0; j<N; ++j){
        float sumExpX = 0;
        for (int i=0; i< nSoftmax; ++i){
            sumExpX += exp(X(i*N+j));
        }
        float sumExpX2 = sumExpX*sumExpX;

        // \sum(dL/dy_j*exp(x_j)
        float dyDotExpX = 0;
        for(int i=0; i< nSoftmax; ++i){
            dyDotExpX += dY(i*N+j)*exp(X(i*N+j));
        }

        for(int i=0; i< nSoftmax; ++i){
            dX(i*N+j) += exp(X(i*N+j))*(dY(i*N+j)*sumExpX-dyDotExpX)/sumExpX2;
        }

    }

#endif

}
void SoftmaxLayer::updateParameters(const float lr, const string& method, const int batchSize){
    //Null
}

int  SoftmaxLayer::getNumParameters(){
    return 0;
}

void SoftmaxLayer::save(const string &netDir) {
//null
}

void SoftmaxLayer::load(const string &netDir) {
//null
}

void SoftmaxLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", "{}", 0, 0, "{}");
}

void SoftmaxLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s,  PrevLayer=%s, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),   m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}

void SoftmaxLayer::initializeLRs(const float lr) {

}

void SoftmaxLayer::updateLRs(const float deltaLoss, const int batchSize) {

}

void SoftmaxLayer::updateParameters(const string &method, const int batchSize) {

}

void SoftmaxLayer::averageParaGradient(const int batchSize) {

}
