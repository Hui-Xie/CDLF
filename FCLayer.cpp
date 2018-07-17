//
// Created by Sheen156 on 6/5/2018.
//

#include "FCLayer.h"
#include "statisTool.h"
#include <iostream>
using namespace std;

// y = W*x+ b
// where y and b is m-D vector, y is output vector;
//       x is n-D input vector
//       W is m*n dimensional matrix
FCLayer::FCLayer(const int id, const string name,const vector<int>& tensorSize, Layer* preLayer):Layer(id, name, tensorSize){
   m_type = "FullyConnected";
   m_n = preLayer->m_tensorSize[0]; //input width
   m_m = m_tensorSize[0];
   setPreviousLayer(preLayer);
   m_pW = new Tensor<float>({m_m,m_n});
   m_pBTensor =  new Tensor<float>({m_m,1});
   m_pdW = new Tensor<float>({m_m,m_n});
   m_pdBTensor =  new Tensor<float>({m_m,1});
}

FCLayer::~FCLayer(){
  if (nullptr != m_pW) delete m_pW;
  if (nullptr != m_pBTensor) delete m_pBTensor;

  if (nullptr != m_pdW) delete m_pdW;
  if (nullptr != m_pdBTensor) delete m_pdBTensor;
}

void FCLayer::initialize(const string& initialMethod)
{
    if ("Xavier" == initialMethod) {
        xavierInitialize(m_pW);
        long nRow = m_pBTensor->getDims()[0];
        const float sigmaB = 1.0 / nRow;
        generateGaussian(m_pBTensor, 0, sigmaB);
    }
    else{
        cout<<"Error: Initialize Error in FCLayer."<<endl;

    }
}

void FCLayer::forward(){
    *m_pYTensor = (*m_pW) * (*(m_prevLayerPointer->m_pYTensor)) + *m_pBTensor;
}

//   y = W*x +b
//  dL/dW = dL/dy * dy/dW = dL/dy * x'
//  dL/db = dL/dy * dy/db = dL/dy
//  dL/dx = dL/dy * dy/dx = W' * dL/dy
void FCLayer::backward(){
    Tensor<float>& dLdy = *m_pdYTensor;
    *m_pdW = dLdy * trans(*(m_prevLayerPointer->m_pYTensor));
    *m_pdBTensor = dLdy;
    *(m_prevLayerPointer->m_pdYTensor) = trans(*m_pW) * dLdy;
}

void FCLayer::updateParameters(const float lr, const string& method) {
    if ("sgd" == method){
        *m_pW -= lr* (*m_pdW);
        *m_pBTensor -= lr* (*m_pdBTensor);
    }
}

void FCLayer::printWandBVector(){
    cout<<"LayerType: "<<m_type <<"; MatrixSize "<<m_m<<"*"<<m_n<<"; W: "<<endl;
    cout<<*m_pW<<endl;
    cout<<"B-transpose:"<<trans(*m_pBTensor)<<endl;
}

void FCLayer::printdWanddBVector(){
    cout<<"LayerType: "<<m_type <<"; MatrixSize "<<m_m<<"*"<<m_n<<"; dW: "<<endl;
    cout<<*m_pdW<<endl;
    cout<<"dB-transpose:"<<trans(*m_pdBTensor)<<endl;

}
