//
// Created by Hui Xie on 6/7/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#include <ReLU.h>

#ifdef Use_GPU
#include <CudnnActivation.h>
#endif

//ReLU has just one previous layer.

ReLU::ReLU(const int id, const string& name,Layer* prevLayer, const vector<int>& tensorSize, const float k): Layer(id,name, tensorSize){
    m_type = "ReLU";
    m_k = k;
    addPreviousLayer(prevLayer);

    if (length(m_tensorSize) != length(m_prevLayer->m_tensorSize)){
        cout<<"Error: The output TensorSize does not equal with the one of the previous layer in ReLU construction at  layID = "<<id<<endl;
    }
}

ReLU::~ReLU(){

}

// Y = X if X >= m_k;
// Y = 0 if x >  m_k;
// dL/dx = dL/dy * dy/dx = dL/dy if X>=m_k;
// dL/dx = 0 if X < m_k
void ReLU::forward(){
#ifdef Use_GPU
    CudnnActivation cudnnActivation(this, CUDNN_ACTIVATION_RELU);
    cudnnActivation.forward();
#else

    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    const int N = Y.getLength();
    for (int i=0; i< N; ++i){
       if (X.e(i) >= m_k ) Y.e(i) = X.e(i);
       else Y.e(i) = 0;
    }
#endif
}

void ReLU::backward(bool computeW, bool computeX) {
#ifdef Use_GPU
    if (computeX) {
        CudnnActivation cudnnActivation(this, CUDNN_ACTIVATION_RELU);
        cudnnActivation.backward(computeW, computeX);
    }
#else

    if (computeX){
        Tensor<float>& dY = *m_pdYTensor;
        Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
        Tensor<float>& X = *m_prevLayer->m_pYTensor;
        const int N = dY.getLength();
        for(int i=0; i< N; ++i){
            if (X.e(i) >= m_k) dX.e(i) += dY.e(i);
            // all dX.e(i) = 0 in zeroDYTensor() method in each iteration.
        }
    }
#endif

}
void ReLU::initialize(const string& initialMethod){
    //null
}

void ReLU::zeroParaGradient(){
    //null
}

void ReLU::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}

int  ReLU::getNumParameters(){
    return 0;
}

void ReLU::save(const string &netDir) {
   //null
}

void ReLU::load(const string &netDir) {
  //null
}

void ReLU::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %f, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", "{}", 0, m_k, "{}");
}

void ReLU::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, k=%f, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),  m_prevLayer->m_name.c_str(), m_k, vector2Str(m_tensorSize).c_str());
}

void ReLU::initializeLRs(const float lr) {

}

void ReLU::updateLRs(const float deltaLoss, const int batchSize) {

}

void ReLU::updateParameters(const string &method, const int batchSize) {

}

void ReLU::averageParaGradient(const int batchSize) {

}
