
#include <CudnnSoftmax.h>


CudnnSoftmax::CudnnSoftmax(Layer *pLayer): Cudnn(pLayer)
{
    m_algorithm = CUDNN_SOFTMAX_FAST;
    m_mode = CUDNN_SOFTMAX_MODE_CHANNEL;

    setDescriptors();
}


CudnnSoftmax::~CudnnSoftmax(){
   //null;
}

void CudnnSoftmax::setXDescriptor() {
    vector<int> & tensorSize = m_pLayer->m_prevLayer->m_tensorSize;

    //The first dimension of the tensor defines the batch size n, and the second dimension defines the number of features maps c.
    //Softmax over channel feature.
    int nbDims = tensorSize.size()+1;

    int* dimA = new int[nbDims];
    dimA[0] = 1;
    for (int i=1; i< nbDims; ++i){
        dimA[i]  = tensorSize[i-1];
    }

    int* strideA = new int [nbDims];  //span in each dimension. It is a different concept with filter-stride in convolution.
    dimA2SpanA(dimA, nbDims, strideA);

    checkCUDNN(cudnnSetTensorNdDescriptor(m_xDescriptor, CUDNN_DATA_FLOAT, nbDims, dimA, strideA));

    // cout<<"In "<<m_pLayer->m_name<<": ";
    //cout<<"xDescriptor: "<<array2Str(dimA, nbDims)<<endl;

    delete[] dimA;
    delete[] strideA;
}

void CudnnSoftmax::setYDescriptor() {
    vector<int> & tensorSize = m_pLayer->m_tensorSize;

    //The first dimension of the tensor defines the batch size n, and the second dimension defines the number of features maps c.
    int nbDims = tensorSize.size()+1;

    int* dimA = new int[nbDims];
    dimA[0] = 1;
    for (int i=1; i< nbDims; ++i){
        dimA[i]  = tensorSize[i-1];
    }

    int* strideA = new int [nbDims];  //span in each dimension. It is a different concept with filter-stride in convolution.
    dimA2SpanA(dimA, nbDims, strideA);

    checkCUDNN(cudnnSetTensorNdDescriptor(m_yDescriptor, CUDNN_DATA_FLOAT, nbDims, dimA, strideA));

    delete[] dimA;
    delete[] strideA;
}

void CudnnSoftmax::setDescriptors() {
    setXDescriptor();
    setYDescriptor();
}

void CudnnSoftmax::forward() {
    allocateDeviceX();
    allocateDeviceY();
    float alpha = 1;
    float beta = 0;
    checkCUDNN(cudnnSoftmaxForward(m_cudnnContext, m_algorithm, m_mode,
                                   &alpha,
                                   m_xDescriptor, d_pX,
                                   &beta,
                                   m_yDescriptor, d_pY));

    const size_t ySize = length(m_pLayer->m_tensorSize) * sizeof(float);
    cudaMemcpy(m_pLayer->m_pYTensor->getData(), d_pY, ySize, cudaMemcpyDeviceToHost);

    freeDeviceX();
    freeDeviceY();
}

void CudnnSoftmax::backward(bool computeW, bool computeX) {
    allocateDeviceY(true);
    allocateDevicedX();
    allocateDevicedY();
    float alpha = 1;
    float beta = 0;
    checkCUDNN(cudnnSoftmaxBackward(m_cudnnContext, m_algorithm, m_mode,
                                    &alpha,
                                    m_yDescriptor, d_pY,
                                    m_yDescriptor, d_pdY,
                                    &beta,
                                    m_xDescriptor, d_pdX));

    const size_t xSize = length(m_pLayer->m_prevLayer->m_tensorSize)*sizeof(float);
    cudaMemcpy(m_pLayer->m_prevLayer->m_pdYTensor->getData(), d_pdX, xSize, cudaMemcpyDeviceToHost);

    freeDeviceY();
    freeDevicedX();
    freeDevicedY();

}





