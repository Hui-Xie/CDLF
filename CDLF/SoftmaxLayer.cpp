//
// Created by Hui Xie on 7/28/2018.
//

#include "SoftmaxLayer.h"
#include <math.h>       /* exp */

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
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    const int nSoftmax = m_pYTensor->getDims()[0];// a vector's dimension to execute softmax
    const long N = X.getLength()/nSoftmax;  // the number of element vectors needing softmax
    for (long j=0; j<N; ++j){
        float sumExpX = 0;
        for (int i=0; i< nSoftmax; ++i){
            sumExpX += exp(X(i*N+j));
        }
        if (0 == sumExpX){
            cout<<"Error: SoftMax Layer sumExpX ==0 "<<endl;
            sumExpX = 1e-8;
        }
        for (int i=0; i< nSoftmax; ++i){
            Y(i*N+j) = exp(X(i*N+j))/sumExpX;
        }
    }
}

void SoftmaxLayer::backward(){
    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    const int nSoftmax = m_pdYTensor->getDims()[0];// a vector's dimension to execute softmax
    const long N = X.getLength()/nSoftmax;  // the number of element vectors needing softmax

    for (long j=0; j<N; ++j){
        float sumExpX = 0;
        for (int i=0; i< nSoftmax; ++i){
            sumExpX += exp(X(i*N+j));
        }
        if (0 == sumExpX){
            cout<<"Error: SoftMax Layer sumExpX ==0 "<<endl;
            sumExpX = 1e-8;
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
}
void SoftmaxLayer::updateParameters(const float lr, const string& method, const int batchSize){
    //Null
}