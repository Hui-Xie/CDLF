
#include <CudnnBasicConvolution.h>


CudnnBasicConvolution::CudnnBasicConvolution(ConvolutionBasicLayer *pLayer, const vector<int> &filterSize, const int numFilters,
                                                       const int stride): Cudnn(pLayer)
{
    d_pWorkspace = nullptr;
    d_pW = nullptr;
    d_pdW = nullptr;

    m_workspaceSize = 0;
    m_filterSize = filterSize;
    m_numFilters = numFilters;
    m_stride = stride;

    checkCUDNN(cudnnCreateFilterDescriptor(&m_wDescriptor));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&m_convDescriptor));
}


CudnnBasicConvolution::~CudnnBasicConvolution(){
    if (nullptr != d_pWorkspace)  {
        cudaFree(d_pWorkspace);
        d_pWorkspace= nullptr;
    }

    if (nullptr != d_pW)  {
        cudaFree(d_pW);
        d_pW= nullptr;
    }
    if (nullptr != d_pdW)  {
        cudaFree(d_pdW);
        d_pdW= nullptr;
    }

    cudnnDestroyFilterDescriptor(m_wDescriptor);
    cudnnDestroyConvolutionDescriptor(m_convDescriptor);
}

void CudnnBasicConvolution::setConvDescriptor() {
    // That same convolution descriptor can be reused in the backward path provided it corresponds to the same layer.
    const int  arrayLength = m_filterSize.size();  // this arrayLength does not include n, c.
    int* padA = new int [arrayLength];
    int* filterStrideA  = new int [arrayLength];
    int* dilationA = new int[arrayLength];
    for (int i=0; i< arrayLength; ++i){
        padA[i] = 0;
        filterStrideA[i] = m_stride;
        dilationA[i] = 1;
    }

    checkCUDNN(cudnnSetConvolutionNdDescriptor(m_convDescriptor, arrayLength, padA, filterStrideA, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    delete[] padA;
    delete[] filterStrideA;
    delete[] dilationA;
}

void CudnnBasicConvolution::setDescriptors() {
    setXDescriptor();
    setWDescriptor();
    setConvDescriptor();
    setYDescriptor();
}

void CudnnBasicConvolution::allocateDeviceW() {
    const int oneFilterSize = length(m_filterSize)*sizeof(float);
    cudaMalloc(&d_pW, oneFilterSize*m_numFilters);
    for (int i=0; i< m_numFilters; ++i){
        cudaMemcpy(d_pW+i*oneFilterSize, ((ConvolutionBasicLayer*)m_pLayer)->m_pW[i]->getData(), oneFilterSize, cudaMemcpyHostToDevice);
    }
}

//dW can accumulate
void CudnnBasicConvolution::allocateDevicedW() {
    const int oneFilterSize = length(m_filterSize)*sizeof(float);
    cudaMalloc(&d_pdW, oneFilterSize*m_numFilters);
    for (int i=0; i< m_numFilters; ++i){
        cudaMemcpy(d_pdW+i*oneFilterSize, ((ConvolutionBasicLayer*)m_pLayer)->m_pdW[i]->getData(), oneFilterSize, cudaMemcpyHostToDevice);
    }
}

