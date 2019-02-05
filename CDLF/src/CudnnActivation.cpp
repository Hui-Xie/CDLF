
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

void CudnnActivation::setXDescriptor() {
    // in order to support the dimension reshape of activation, setXDescriptor use its output demension
    // because the input and output of actication has same diemnsion.
    vector<int> & tensorSize = m_pLayer->m_tensorSize;

    //The first dimension of the tensor defines the batch size n, and the second dimension defines the number of features maps c.
    int nbDims = tensorSize.size()+2;

    int* dimA = new int[nbDims];
    dimA[0] = 1;
    dimA[1] = 1;
    for (int i=2; i< nbDims; ++i){
        dimA[i]  = tensorSize[i-2];
    }

    int* strideA = new int [nbDims];  //span in each dimension. It is a different concept with filter-stride in convolution.
    dimA2SpanA(dimA, nbDims, strideA);

    checkCUDNN(cudnnSetTensorNdDescriptor(m_xDescriptor, CUDNN_DATA_FLOAT, nbDims, dimA, strideA));

    // cout<<"In "<<m_pLayer->m_name<<": ";
    //cout<<"xDescriptor: "<<array2Str(dimA, nbDims)<<endl;

    delete[] dimA;
    delete[] strideA;
}


void CudnnActivation::setYDescriptor() {
    vector<int> & tensorSize = m_pLayer->m_tensorSize;

    //The first dimension of the tensor defines the batch size n, and the second dimension defines the number of features maps c.
    int nbDims = tensorSize.size()+2;

    int* dimA = new int[nbDims];
    dimA[0] = 1;
    dimA[1] = 1;
    for (int i=2; i< nbDims; ++i){
        dimA[i]  = tensorSize[i-2];
    }

    int* strideA = new int [nbDims];  //span in each dimension. It is a different concept with filter-stride in convolution.
    dimA2SpanA(dimA, nbDims, strideA);

    checkCUDNN(cudnnSetTensorNdDescriptor(m_yDescriptor, CUDNN_DATA_FLOAT, nbDims, dimA, strideA));

    delete[] dimA;
    delete[] strideA;
}



void CudnnActivation::setActivationDescriptor(){
    checkCUDNN(cudnnSetActivationDescriptor(m_activationDescriptor, m_activationMode, CUDNN_PROPAGATE_NAN, 0));
}

void CudnnActivation::setDescriptors() {
    setXDescriptor();
    setYDescriptor();
    setActivationDescriptor();
}



void CudnnActivation::forward() {
    allocateDeviceX();
    allocateDeviceY();
    float alpha = 1;
    float beta = 0;
    checkCUDNN(cudnnActivationForward(m_cudnnContext, m_activationDescriptor,
                                      &alpha,
                                      m_xDescriptor, d_pX,
                                      &beta,
                                      m_yDescriptor, d_pY));

    const size_t ySize = length(m_pLayer->m_tensorSize) * sizeof(float);
    cudaMemcpy(m_pLayer->m_pYTensor->getData(), d_pY, ySize, cudaMemcpyDeviceToHost);

    freeDeviceX();
    freeDeviceY();
}

void CudnnActivation::backward(bool computeW, bool computeX) {
    allocateDeviceX();
    allocateDeviceY(true);
    allocateDevicedX();
    allocateDevicedY();
    float alpha = 1;
    float beta = 0;
    checkCUDNN(cudnnActivationBackward(m_cudnnContext, m_activationDescriptor,
                                       &alpha,
                                       m_yDescriptor, d_pY,
                                       m_yDescriptor, d_pdY,
                                       m_xDescriptor, d_pX,
                                       &beta,
                                       m_xDescriptor, d_pdX));

    const size_t xSize = length(m_pLayer->m_prevLayer->m_tensorSize)*sizeof(float);
    cudaMemcpy(m_pLayer->m_prevLayer->m_pdYTensor->getData(), d_pdX, xSize, cudaMemcpyDeviceToHost);

    freeDeviceX();
    freeDeviceY();
    freeDevicedX();
    freeDevicedY();

}




