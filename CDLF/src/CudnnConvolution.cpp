
#include <CudnnConvolution.h>
#include "ConvolutionLayer.h"

CudnnConvolution::CudnnConvolution(Layer *pLayer, const vector<int> &filterSize, const int numFilters,
                                   const int stride): Cudnn(pLayer, stride)
{
    d_pWorkspace = nullptr;
    d_pX = nullptr;
    d_pY = nullptr;
    d_pFilter = nullptr;
    m_workspaceSize = 0;

    m_filterSize = filterSize;
    m_numFilters = numFilters;
    cudnnCreateFilterDescriptor(&m_filterDescriptor);
    cudnnCreateConvolutionDescriptor(&m_convDescriptor);

    getDimsArrayFromFilterSize(filterSize,m_numFilters, m_filterSizeArray, m_filterArrayDim);

    setConvDescriptorsAndAlgorithm();
}


CudnnConvolution::~CudnnConvolution(){
    if (nullptr != d_pWorkspace)  cudaFree(d_pWorkspace);
    if (nullptr != d_pX)  cudaFree(d_pX);
    if (nullptr != d_pY)  cudaFree(d_pY);
    if (nullptr != d_pFilter)  cudaFree(d_pFilter);

    cudnnDestroyFilterDescriptor(m_filterDescriptor);
    cudnnDestroyConvolutionDescriptor(m_convDescriptor);
    delete m_filterSizeArray;
}

void CudnnConvolution::setConvDescriptorsAndAlgorithm() {
    cudnnSetFilterNdDescriptor(m_filterDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, m_filterArrayDim, m_filterSizeArray);
    cudnnSetConvolution2dDescriptor(m_convDescriptor, 0,0, m_stride, m_stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnGetConvolutionForwardAlgorithm(m_cudnnContext, m_xDescriptor, m_filterDescriptor, m_convDescriptor, m_yDescriptor,
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &m_fwdConvAlgorithm);
    cudnnGetConvolutionForwardWorkspaceSize(m_cudnnContext, m_xDescriptor, m_filterDescriptor, m_convDescriptor, m_yDescriptor,
                    m_fwdConvAlgorithm, &m_workspaceSize);
}

void CudnnConvolution::allocateDeviceMemAndCopy() {
    cudaMalloc(&d_pWorkspace, m_workspaceSize);

    const int xSize = length(m_pLayer->m_prevLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pX, xSize);
    cudaMemcpy(d_pX, m_pLayer->m_prevLayer->m_pYTensor->getData(), xSize, cudaMemcpyHostToDevice);

    const int ySize = length(m_pLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pY, ySize);
    cudaMemset(d_pY, 0, ySize);


    const int oneFilterSize = length(m_filterSize)*sizeof(float);
    cudaMalloc(&d_pFilter, oneFilterSize*m_numFilters);
    for (int i=0; i< m_numFilters; ++i){
        cudaMemcpy(d_pFilter+i*oneFilterSize, ((ConvolutionLayer*)m_pLayer)->m_pW[i]->getData(), oneFilterSize, cudaMemcpyHostToDevice);
    }
}
