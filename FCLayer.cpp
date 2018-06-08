//
// Created by Sheen156 on 6/5/2018.
//

#include "FCLayer.h"
#include "statisTool.h"

// y = W*x+ b
// where y and b is m-D vector, y is output vector;
//       x is n-D input vector
//       W is m*n dimensional matrix
FCLayer::FCLayer(const long width, Layer* preLayer):Layer(width){
   m_type = "FullyConnected";
   m_n = preLayer->m_width; //input width
   m_m = m_width;

   m_prevLayerPointer = preLayer;
   m_prevLayerPointer->m_nextLayerPointer = this;

   m_pYVector = new DynamicVector<float>(m_m);
   m_pW = new DynamicMatrix<float>(m_m,m_n);
   m_pBVector =  new DynamicVector<float>(m_m);

   m_pdYVector = new DynamicVector<float>(m_m);
   m_pdW = new DynamicMatrix<float>(m_m,m_n);
   m_pdBVector =  new DynamicVector<float>(m_m);
}

FCLayer::~FCLayer(){
  if (nullptr != m_pYVector) delete m_pYVector;
  if (nullptr != m_pW) delete m_pW;
  if (nullptr != m_pBVector) delete m_pBVector;

  if (nullptr != m_pdYVector) delete m_pYVector;
  if (nullptr != m_pdW) delete m_pW;
  if (nullptr != m_pdBVector) delete m_pBVector;
}

void FCLayer::initialize(const string& initialMethod)
{
    if ("Xavier" == initialMethod) {
        xavierInitialize(m_pW);
        long nRow = m_pBVector->size();
        const float sigmaB = 1.0 / nRow;
        generateGaussian(m_pBVector, 0, sigmaB);
    }

}

void FCLayer::forward(){
    *m_pYVector = (*m_pW) * (*(m_prevLayerPointer->m_pYVector)) + *m_pBVector;
}

//   y = W*x +b
//  dL/dW = dL/dy * dy/dW = dL/dy * x'
//  dL/db = dL/dy * dy/db = dL/dy
//  dL/dx = dL/dy * dy/dx = W' * dL/dy
void FCLayer::backward(){
    DynamicVector<float>& dLdy = *m_pdYVector;
    *m_pdW = dLdy * trans(*(m_prevLayerPointer->m_pYVector));
    *m_pdBVector = dLdy;
    *(m_prevLayerPointer->m_pdYVector) = trans(*m_pW) * dLdy;
}
