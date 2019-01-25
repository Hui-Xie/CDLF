
#include <CudnnConvolution.h>


CudnnConvolution::CudnnConvolution(ConvolutionLayer *pLayer, const vector<int> &filterSize, const int numFilters,
                                   const int stride): Cudnn(pLayer)
{
    d_pWorkspace = nullptr;
    d_pX = nullptr;
    d_pY = nullptr;
    d_pW = nullptr;
    d_pdX = nullptr;
    d_pdY = nullptr;
    d_pdW = nullptr;
    m_workspaceSize = 0;

    m_filterSize = filterSize;
    m_numFilters = numFilters;
    m_stride = stride;

    checkCUDNN(cudnnCreateFilterDescriptor(&m_wDescriptor));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&m_convDescriptor));

    setDescriptors();

}


CudnnConvolution::~CudnnConvolution(){
    if (nullptr != d_pWorkspace)  cudaFree(d_pWorkspace);
    if (nullptr != d_pX)  cudaFree(d_pX);
    if (nullptr != d_pY)  cudaFree(d_pY);
    if (nullptr != d_pW)  cudaFree(d_pW);
    if (nullptr != d_pdX)  cudaFree(d_pdX);
    if (nullptr != d_pdY)  cudaFree(d_pdY);
    if (nullptr != d_pdW)  cudaFree(d_pdW);

    cudnnDestroyFilterDescriptor(m_wDescriptor);
    cudnnDestroyConvolutionDescriptor(m_convDescriptor);
}

void CudnnConvolution::forward() {
    allocateDeviceX();
    allocateDeviceW();
    allocateDeviceY();
    setForwardAlg();
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(m_cudnnContext, m_xDescriptor, m_wDescriptor, m_convDescriptor, m_yDescriptor,
                                                       m_fwdAlg, &m_workspaceSize));
    cudaMalloc(&d_pWorkspace, m_workspaceSize);

    float alpha = 1;
    float beta = 0;
    checkCUDNN(cudnnConvolutionForward(m_cudnnContext, &alpha,
                            m_xDescriptor, d_pX,
                            m_wDescriptor, d_pW,
                            m_convDescriptor, m_fwdAlg,
                            d_pWorkspace, m_workspaceSize,
                            &beta,
                            m_yDescriptor, d_pY));
    const int ySize = length(m_pLayer->m_tensorSize)*sizeof(float);
    cudaMemcpy(m_pLayer->m_pYTensor->getData(), d_pY, ySize, cudaMemcpyDeviceToHost);
}

void CudnnConvolution::backward(bool computeW, bool computeX) {
    allocateDevicedY();

    if (computeW){
        allocateDeviceX();
        allocateDevicedW();
        setBackWardFilterAlg();
        checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(m_cudnnContext, m_xDescriptor, m_yDescriptor, m_convDescriptor, m_wDescriptor,
                                                                m_bwdFilterAlg, &m_workspaceSize));
        cudaMalloc(&d_pWorkspace, m_workspaceSize);

        float alpha = 1;
        float beta = 1;
        checkCUDNN(cudnnConvolutionBackwardFilter(m_cudnnContext, &alpha,
                                                m_xDescriptor, d_pX,
                                                m_yDescriptor, d_pdY,
                                                m_convDescriptor, m_bwdFilterAlg,
                                                d_pWorkspace, m_workspaceSize,
                                                &beta,
                                                m_wDescriptor, d_pdW));

        const int wSize = length(m_filterSize);
        for (int i=0; i< m_numFilters; ++i){
            cudaMemcpy(((ConvolutionLayer*)m_pLayer)->m_pdW[i]->getData(), d_pdW+i*wSize, wSize* sizeof(float), cudaMemcpyDeviceToHost);
        }

        if (nullptr != d_pWorkspace){
            cudaFree(d_pWorkspace);
            d_pWorkspace = nullptr;
        }
    }

    if (computeX){
        allocateDevicedX();
        allocateDeviceW();
        setBackwardDataAlg();
        checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(m_cudnnContext, m_wDescriptor, m_yDescriptor, m_convDescriptor, m_xDescriptor,
                                                           m_bwdDataAlg, &m_workspaceSize));
        cudaMalloc(&d_pWorkspace, m_workspaceSize);

        float alpha = 1;
        float beta = 0;
        checkCUDNN(cudnnConvolutionBackwardData(m_cudnnContext, &alpha,
                                           m_wDescriptor, d_pW,
                                           m_yDescriptor, d_pdY,
                                           m_convDescriptor, m_bwdDataAlg,
                                           d_pWorkspace, m_workspaceSize,
                                           &beta,
                                           m_xDescriptor, d_pdX));
        const int xSize = length(m_pLayer->m_tensorSize)*sizeof(float);
        cudaMemcpy(m_pLayer->m_pdYTensor->getData(), d_pdX, xSize, cudaMemcpyDeviceToHost);

        if (nullptr != d_pWorkspace){
            cudaFree(d_pWorkspace);
            d_pWorkspace = nullptr;
        }
    }
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

    checkCUDNN(cudnnSetFilterNdDescriptor(m_wDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, nbDims, filterDimA));

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
    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(m_convDescriptor, m_xDescriptor, m_wDescriptor, nbDims, tensorOuputDimA));
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
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(m_cudnnContext, m_xDescriptor, m_wDescriptor, m_convDescriptor, m_yDescriptor,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &m_fwdAlg));
}

void CudnnConvolution::setBackwardDataAlg(){
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(m_cudnnContext, m_wDescriptor, m_yDescriptor, m_convDescriptor, m_xDescriptor,
                                                  CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &m_bwdDataAlg));
}
void CudnnConvolution::setBackWardFilterAlg(){
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(m_cudnnContext, m_xDescriptor, m_yDescriptor, m_convDescriptor, m_wDescriptor,
                                                 CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &m_bwdFilterAlg));
}

void CudnnConvolution::setDescriptors() {
    setXDescriptor();
    setFilterDescriptor();
    setConvDescriptor();
    setYDescriptor();

}

void CudnnConvolution::allocateDeviceX() {
    const int xSize = length(m_pLayer->m_prevLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pX, xSize);
    cudaMemcpy(d_pX, m_pLayer->m_prevLayer->m_pYTensor->getData(), xSize, cudaMemcpyHostToDevice);
}

void CudnnConvolution::allocateDeviceY() {
    const int ySize = length(m_pLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pY, ySize);
    cudaMemset(d_pY, 0, ySize);
}

void CudnnConvolution::allocateDeviceW() {
    const int oneFilterSize = length(m_filterSize)*sizeof(float);
    cudaMalloc(&d_pW, oneFilterSize*m_numFilters);
    for (int i=0; i< m_numFilters; ++i){
        cudaMemcpy(d_pW+i*oneFilterSize, ((ConvolutionLayer*)m_pLayer)->m_pW[i]->getData(), oneFilterSize, cudaMemcpyHostToDevice);
    }
}

void CudnnConvolution::allocateDevicedX() {
    const int xSize = length(m_pLayer->m_prevLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pdX, xSize);
    cudaMemset(d_pdX, 0, xSize);
}

void CudnnConvolution::allocateDevicedY() {
    const int ySize = length(m_pLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pdY, ySize);
    cudaMemcpy(d_pdY, m_pLayer->m_pdYTensor->getData(), ySize, cudaMemcpyHostToDevice);
}

//dW can accumulate
void CudnnConvolution::allocateDevicedW() {
    const int oneFilterSize = length(m_filterSize)*sizeof(float);
    cudaMalloc(&d_pdW, oneFilterSize*m_numFilters);
    for (int i=0; i< m_numFilters; ++i){
        cudaMemcpy(d_pdW+i*oneFilterSize, ((ConvolutionLayer*)m_pLayer)->m_pdW[i]->getData(), oneFilterSize, cudaMemcpyHostToDevice);
    }
}

