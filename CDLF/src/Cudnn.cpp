
#include <Cudnn.h>

#include "Cudnn.h"

Cudnn::Cudnn(Layer* pLayer){
    m_pLayer = pLayer;
    checkCUDNN(cudnnCreate(&m_cudnnContext));
    checkCUDNN(cudnnCreateTensorDescriptor(&m_xDescriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&m_yDescriptor));


}

Cudnn::~Cudnn(){
   cudnnDestroyTensorDescriptor(m_xDescriptor);
   cudnnDestroyTensorDescriptor(m_yDescriptor);
   cudnnDestroy(m_cudnnContext);
}



