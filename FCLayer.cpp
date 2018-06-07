//
// Created by Sheen156 on 6/5/2018.
//

#include "FCLayer.h"

// y = W*x+ b
// where y and b is m-D vector, y is output vector;
//       x is n-D input vector
//       W is m*n dimensional matrix
FCLayer::FCLayer(long width, Layer* preLayer):Layer(width){
   m_type = "FullyConnected";
   m_n = preLayer->m_width; //input width
   m_m = m_width;
   m_pYVector = new DynamicVector<float>(m_m);
   m_pW = new DynamicMatrix<float>(m_m,m_n);
   m_pBVector =  new DynamicVector<float>(m_m);
}

FCLayer::~FCLayer(){
  if (nullptr != m_pYVector) delete m_pYVector;
  if (nullptr != m_pW) delete m_pW;
  if (nullptr != m_pBVector) delete m_pBVector;
}

void FCLayer::initialize(const string& initialMethod)
{
    if ("Xavier" != initialMethod) return;
    //implement a Xavier initialize method.


}

void FCLayer::forward(){
    *m_pYVector = (*m_pW) * (*(m_preLayerPointer->m_pYVector)) + *m_pBVector;
}

void FCLayer::backward(){

}
