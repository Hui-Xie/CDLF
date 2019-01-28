
#include <CudnnBasicConvolution.h>


CudnnBasicConvolution::CudnnBasicConvolution(ConvolutionBasicLayer *pLayer, const vector<int> &filterSize, const int numFilters,
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


CudnnBasicConvolution::~CudnnBasicConvolution(){
    if (nullptr != d_pWorkspace)  {
        cudaFree(d_pWorkspace);
        d_pWorkspace= nullptr;

    }
    if (nullptr != d_pX)  {
        cudaFree(d_pX);
        d_pX= nullptr;
    }
    if (nullptr != d_pY)  {
        cudaFree(d_pY);
        d_pY= nullptr;
    }
    if (nullptr != d_pW)  {
        cudaFree(d_pW);
        d_pW= nullptr;
    }
    if (nullptr != d_pdX)  {
        cudaFree(d_pdX);
        d_pdX= nullptr;
    }
    if (nullptr != d_pdY)  {
        cudaFree(d_pdY);
        d_pdY = nullptr;
    }
    if (nullptr != d_pdW)  {
        cudaFree(d_pdW);
        d_pdW= nullptr;
    }

    cudnnDestroyFilterDescriptor(m_wDescriptor);
    cudnnDestroyConvolutionDescriptor(m_convDescriptor);
}


void CudnnBasicConvolution::setXDescriptor() {
    vector<int> & tensorSize = m_pLayer->m_prevLayer->m_tensorSize;

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

    delete[] dimA;
    delete[] strideA;
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

void CudnnBasicConvolution::allocateDeviceX() {
    const int xSize = length(m_pLayer->m_prevLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pX, xSize);
    cudaMemcpy(d_pX, m_pLayer->m_prevLayer->m_pYTensor->getData(), xSize, cudaMemcpyHostToDevice);
}

void CudnnBasicConvolution::allocateDeviceY() {
    const int ySize = length(m_pLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pY, ySize);
    cudaMemset(d_pY, 0, ySize);
}

void CudnnBasicConvolution::allocateDeviceW() {
    const int oneFilterSize = length(m_filterSize)*sizeof(float);
    cudaMalloc(&d_pW, oneFilterSize*m_numFilters);
    for (int i=0; i< m_numFilters; ++i){
        cudaMemcpy(d_pW+i*oneFilterSize, ((ConvolutionBasicLayer*)m_pLayer)->m_pW[i]->getData(), oneFilterSize, cudaMemcpyHostToDevice);
    }
}

void CudnnBasicConvolution::allocateDevicedX() {
    const int xSize = length(m_pLayer->m_prevLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pdX, xSize);
    cudaMemset(d_pdX, 0, xSize);
}

void CudnnBasicConvolution::allocateDevicedY() {
    const int ySize = length(m_pLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pdY, ySize);
    cudaMemcpy(d_pdY, m_pLayer->m_pdYTensor->getData(), ySize, cudaMemcpyHostToDevice);
}

//dW can accumulate
void CudnnBasicConvolution::allocateDevicedW() {
    const int oneFilterSize = length(m_filterSize)*sizeof(float);
    cudaMalloc(&d_pdW, oneFilterSize*m_numFilters);
    for (int i=0; i< m_numFilters; ++i){
        cudaMemcpy(d_pdW+i*oneFilterSize, ((ConvolutionBasicLayer*)m_pLayer)->m_pdW[i]->getData(), oneFilterSize, cudaMemcpyHostToDevice);
    }
}

