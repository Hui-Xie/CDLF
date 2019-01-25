
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
    delete[] m_filterSizeArray;
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

void CudnnConvolution::setFilterDescriptor() {
    int k = m_numFilters;
    int c = 1, h =1, w =1;
    if (2 == m_filterSize.size()){
        h = m_filterSize[0];
        w = m_filterSize[1];
    }
    else if (3 ==m_filterSize.size()){
        c = m_filterSize[0];
        h = m_filterSize[1];
        w = m_filterSize[2];
    }
    else{
        cout<<"Error: cudnn can not support 4D or above filter in layer "<<m_pLayer->m_name<<endl;
        std::exit(EXIT_FAILURE);
    }

    checkCUDNN(cudnnSetFilter4dDescriptor(m_filterDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w));
}

void CudnnConvolution::setConvDescriptor() {
    // That same convolution descriptor can be reused in the backward path provided it corresponds to the same layer.
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

    delete[] padA;
    delete[] filterStrideA;
    delete[] dilationA;

}

void CudnnConvolution::setYDescriptor() {
    int nbDim = m_filterSize.size()+2;
    int* tensorOuputDimA = new int [nbDim];
    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(m_convDescriptor, m_xDescriptor, m_filterDescriptor, nbDim, tensorOuputDimA));
    if (length(nbDim, tensorOuputDimA) != length(m_pLayer->m_tensorSize)){
        printf("In Convolution layer: %s\n", m_pLayer->m_name.c_str());
        printf("m_tensorSize = %s\n", vector2Str(m_pLayer->m_tensorSize).c_str());
        printf("cudnnComputedOutputDims  = %s\n",  array2Str(tensorOuputDimA, nbDim).c_str());
        std::exit(EXIT_FAILURE);
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
    setFilterDescriptor();
    setConvDescriptor();
    setYDescriptor();
    setForwardAlg();
}

