
#include <CudnnConvolution.h>


CudnnConvolution::CudnnConvolution(ConvolutionLayer *pLayer, const vector<int> &filterSize, const int numFilters,
                                   const int stride): Cudnn(pLayer)
{
    d_pWorkspace = nullptr;
    d_pX = nullptr;
    d_pY = nullptr;
    d_pFilter = nullptr;
    m_workspaceSize = 0;

    m_filterSize = filterSize;
    m_numFilters = numFilters;
    m_stride = stride;

    checkCUDNN(cudnnCreateFilterDescriptor(&m_filterDescriptor));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&m_convDescriptor));

    setDescriptorsAndAlg();

}


CudnnConvolution::~CudnnConvolution(){
    if (nullptr != d_pWorkspace)  cudaFree(d_pWorkspace);
    if (nullptr != d_pX)  cudaFree(d_pX);
    if (nullptr != d_pY)  cudaFree(d_pY);
    if (nullptr != d_pFilter)  cudaFree(d_pFilter);

    cudnnDestroyFilterDescriptor(m_filterDescriptor);
    cudnnDestroyConvolutionDescriptor(m_convDescriptor);
}

void CudnnConvolution::forward() {
    allocateDeviceMem();
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



void CudnnConvolution::setXDescriptor() {
    vector<int> & tensorSize = m_pLayer->m_prevLayer->m_tensorSize;

    //The first dimension of the tensor defines the batch size n, and the second dimension defines the number of features maps c.
    int nbDims = tensorSize.size()+2;

    int* dimA = new int[nbDims];
    int* strideA = new int [nbDims];
    for (int i=0; i< nbDims; ++i){
        if (i< 2){
           dimA[i]  = 1;
           strideA[i]  =1;
        }
        else{
            dimA[i]  = tensorSize[i-2];
            strideA[i]  = m_stride;
        }
    }

    checkCUDNN(cudnnSetTensorNdDescriptor(m_xDescriptor, CUDNN_DATA_FLOAT, nbDims, dimA, strideA));

    delete[] dimA;
    delete[] strideA;
}

void CudnnConvolution::setFilterDescriptor() {
    //The first dimension of the tensor defines the batch size n, and the second dimension defines the number of features maps c.
    int nbDims = m_filterSize.size()+2;

    int* filterDimA = new int[nbDims];
    filterDimA[0] = m_numFilters;
    filterDimA[1] = 1;
    for (int i=2; i< nbDims; ++i){
        filterDimA[i]  = m_filterSize[i-2];
    }

    checkCUDNN(cudnnSetFilterNdDescriptor(m_filterDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, nbDims, filterDimA));

    delete[] filterDimA;
}

void CudnnConvolution::setConvDescriptor() {
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



void CudnnConvolution::setYDescriptor() {
    int nbDims = m_filterSize.size()+2;
    int* tensorOuputDimA = new int [nbDims];
    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(m_convDescriptor, m_xDescriptor, m_filterDescriptor, nbDims, tensorOuputDimA));
    if (length(nbDims, tensorOuputDimA) != length(m_pLayer->m_tensorSize)){
        printf("In Convolution layer: %s\n", m_pLayer->m_name.c_str());
        printf("m_tensorSize = %s\n", vector2Str(m_pLayer->m_tensorSize).c_str());
        printf("cudnnComputedOutputDims  = %s\n",  array2Str(tensorOuputDimA, nbDims).c_str());
        std::exit(EXIT_FAILURE);
    }
    else{
        int* strideA = new int[nbDims];
        for (int i=0; i< nbDims; ++i){
            strideA[i] = 1;
        }
        checkCUDNN(cudnnSetTensorNdDescriptor(m_yDescriptor, CUDNN_DATA_FLOAT, nbDims, tensorOuputDimA, strideA));
        delete[] strideA;
    }

    delete[] tensorOuputDimA;
}

void CudnnConvolution::setForwardAlg() {
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(m_cudnnContext, m_xDescriptor, m_filterDescriptor, m_convDescriptor, m_yDescriptor,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &m_fwdConvAlgorithm));
}

void CudnnConvolution::allocateDeviceMem() {
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(m_cudnnContext, m_xDescriptor, m_filterDescriptor, m_convDescriptor, m_yDescriptor,
                                                       m_fwdConvAlgorithm, &m_workspaceSize));
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

void CudnnConvolution::setDescriptorsAndAlg() {
    setXDescriptor();
    setFilterDescriptor();
    setConvDescriptor();
    setYDescriptor();
    setForwardAlg();
}

