
#include <CudnnActivation.h>


CudnnActivation::CudnnActivation(Layer *pLayer, cudnnActivationMode_t activationMode): Cudnn(pLayer)
{
    m_activationMode = activationMode;
    checkCUDNN(cudnnCreateActivationDescriptor(&m_activationDescriptor));

    setDescriptors();
}


CudnnActivation::~CudnnActivation(){
    cudnnDestroyActivationDescriptor(m_activationDescriptor);
}

void CudnnActivation::setYDescriptor() {

}



void CudnnActivation::setActivationDescriptor(){

}

void CudnnActivation::setDescriptors() {
    setXDescriptor();
    setYDescriptor();
    setActivationDescriptor();
}



void CudnnActivation::forward() {

}

void CudnnActivation::backward(bool computeW, bool computeX) {

}



