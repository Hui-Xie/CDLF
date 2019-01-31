//
// Created by Hui Xie on 8/4/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <SigmoidLayer.h>
#ifdef Use_GPU
   #include <CudnnActivation.h>
#endif

SigmoidLayer::SigmoidLayer(const int id, const string& name,Layer* prevLayer, const vector<int>& tensorSize, const int k): Layer(id,name, tensorSize){
    m_type = "SigmoidLayer";
    m_k = k;
    addPreviousLayer(prevLayer);

    if (length(m_tensorSize) != length(m_prevLayer->m_tensorSize)){
        cout<<"Error: The output TensorSize does not equal with the one of the previous layer in ReLU construction at  layID = "<<id<<endl;
    }

}

SigmoidLayer::~SigmoidLayer(){
  //null
}

/* Y = k/( 1+ exp(-x)) in element-wise
 * dL/dx = dL/dY * dY/dx = dL/dY *k * exp(x)/(1 +exp(x))^2
 * dL/dx = dL/dY * dY/dx = dL/dy *Y*(1- Y/k);
 * */
void SigmoidLayer::forward(){
#ifdef Use_GPU
    CudnnActivation cudnnActivation(this, CUDNN_ACTIVATION_SIGMOID);
    cudnnActivation.forward();
#else
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    const int N = Y.getLength();
    for (int i=0; i< N; ++i){
        float exp_x = exp(-X.e(i));
        Y.e(i) = m_k/(1+exp_x);
    }
#endif

}
void SigmoidLayer::backward(bool computeW, bool computeX){
#ifdef Use_GPU
    if (computeX) {
        CudnnActivation cudnnActivation(this, CUDNN_ACTIVATION_SIGMOID);
        cudnnActivation.backward(computeW, computeX);
    }
#else

    if (computeX){
        Tensor<float>& Y = *m_pYTensor;
        Tensor<float>& dY = *m_pdYTensor;
        Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
        const int N = dY.getLength();

        // use method: dL/dx = dL/dY * dY/dx = dL/dY *k * exp(x)/(1 +exp(x))^2
        /*Tensor<float>& X = *m_prevLayer->m_pYTensor;
        for(int i=0; i< N; ++i){
            float  expx = exp(X.e(i));
            dX.e(i) += dY.e(i)*m_k*expx/pow(1+expx,2);
        }*/

        //use method:dL/dx = dL/dY * dY/dx = dL/dy *Y*(1- Y/k);
        for(int i=0; i< N; ++i){
            dX.e(i) += dY.e(i)*Y.e(i)*(1-Y.e(i)/m_k);
        }
    }
#endif

}
void SigmoidLayer::initialize(const string& initialMethod){
    //null
}

void SigmoidLayer::zeroParaGradient(){
    //null
}

void SigmoidLayer::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}

int  SigmoidLayer::getNumParameters(){
    return 0;
}

void SigmoidLayer::save(const string &netDir) {
  //null
}

void SigmoidLayer::load(const string &netDir) {
  //null
}

void SigmoidLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", "{}", 0, m_k, "{}");
}

void SigmoidLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, k=%d, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),  m_prevLayer->m_name.c_str(), m_k, vector2Str(m_tensorSize).c_str());
}
