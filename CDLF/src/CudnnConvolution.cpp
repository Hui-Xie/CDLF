
#include <CudnnConvolution.h>


CudnnConvolution::CudnnConvolution(ConvolutionLayer *pLayer, const vector<int> &filterSize, const int numFilters,
                                   const int stride): Cudnn(pLayer, stride)
{
    d_pWorkspace = nullptr;
    d_pX = nullptr;
    d_pY = nullptr;
    d_pFilter = nullptr;
    m_workspaceSize = 0;

    m_filterSize = filterSize;
    m_numFilters = numFilters;
    checkCUDNN(cudnnCreateFilterDescriptor(&m_filterDescriptor));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&m_convDescriptor));

    getDimsArrayFromFilterSize(filterSize,m_numFilters, m_filterSizeArray, m_filterArrayDim);

    setConvDescriptorsAndAlgorithm();
    isOutputDimCorrect();
}


CudnnConvolution::~CudnnConvolution(){
    if (nullptr != d_pWorkspace)  cudaFree(d_pWorkspace);
    if (nullptr != d_pX)  cudaFree(d_pX);
    if (nullptr != d_pY)  cudaFree(d_pY);
    if (nullptr != d_pFilter)  cudaFree(d_pFilter);

    cudnnDestroyFilterDescriptor(m_filterDescriptor);
    cudnnDestroyConvolutionDescriptor(m_convDescriptor);
    delete[] m_filterSizeArray;
}

void CudnnConvolution::setConvDescriptorsAndAlgorithm() {

    checkCUDNN(cudnnSetFilterNdDescriptor(m_filterDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, m_filterArrayDim, m_filterSizeArray));

    const int  arrayLength = m_filterSize.size();
    int* padA = new int [arrayLength];
    int* filterStrideA  = new int [arrayLength];
    int* dilationA = new int[arrayLength];
    for (int i=0; i< arrayLength; ++i){
        padA[i] = 0;
        filterStrideA[i] = m_stride;
        dilationA[i] = 1;
    }
    checkCUDNN(cudnnSetConvolutionNdDescriptor(m_convDescriptor, arrayLength, padA, filterStrideA, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(m_cudnnContext, m_xDescriptor, m_filterDescriptor, m_convDescriptor, m_yDescriptor,
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &m_fwdConvAlgorithm));

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(m_cudnnContext, m_xDescriptor, m_filterDescriptor, m_convDescriptor, m_yDescriptor,
                    m_fwdConvAlgorithm, &m_workspaceSize));

    delete[] padA;
    delete[] filterStrideA;
    delete[] dilationA;
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

void CudnnConvolution::forward() {
    allocateDeviceMemAndCopy();
    float alpha = 1;
    float beta = 0;
    checkCUDNN(cudnnConvolutionForward(m_cudnnContext, &alpha,
                            m_xDescriptor, d_pX,
                            m_filterDescriptor, d_pFilter,
                            m_convDescriptor, m_fwdConvAlgorithm,
                            d_pWorkspace, m_workspaceSize,
                            &beta,
                            m_yDescriptor, d_pY));
    const int ySize = length(m_pLayer->m_tensorSize)*sizeof(float);
    cudaMemcpy(m_pLayer->m_pYTensor->getData(), d_pY, ySize, cudaMemcpyDeviceToHost);
}

void CudnnConvolution::backward() {

}

bool CudnnConvolution::isOutputDimCorrect() {
    int nbDim = ((ConvolutionLayer*)m_pLayer)->m_filterSize.size();
    nbDim = nbDim+2;
    int* tensorOuputDimA = new int [nbDim];
    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(m_convDescriptor, m_xDescriptor, m_filterDescriptor, nbDim, tensorOuputDimA));
    bool result  = true;
    for (int i = 0; i<nbDim; ++i){
       if (((ConvolutionLayer*)m_pLayer)->m_tensorSizeBeforeCollapse[i] != tensorOuputDimA[i]){
           result = false;
           break;
       }
    }

    if (!result){
       printf("In Convolution layer: %s\n", m_pLayer->m_name.c_str());
       printf("m_tensorSizeBeforeCollapse = %s\n", vector2Str(((ConvolutionLayer*)m_pLayer)->m_tensorSizeBeforeCollapse).c_str());
       printf("cudnnComputedOutputDims  = %s\n",  array2Str(tensorOuputDimA, nbDim).c_str());
    }

    delete[] tensorOuputDimA;
    return result;
}
