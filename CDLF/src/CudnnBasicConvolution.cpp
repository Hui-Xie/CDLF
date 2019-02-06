
#include <CudnnBasicConvolution.h>


CudnnBasicConvolution::CudnnBasicConvolution(ConvolutionBasicLayer *pLayer): Cudnn(pLayer)
{
    d_pWorkspace = nullptr;
    d_pW = nullptr;
    d_pdW = nullptr;
    m_workspaceSize = 0;

    checkCUDNN(cudnnCreateFilterDescriptor(&m_wDescriptor));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&m_convDescriptor));
}


CudnnBasicConvolution::~CudnnBasicConvolution(){
    freeWorkSpace();
    freeDeviceW();
    freeDevicedW();

    cudnnDestroyFilterDescriptor(m_wDescriptor);
    cudnnDestroyConvolutionDescriptor(m_convDescriptor);
}

void CudnnBasicConvolution::setXDescriptor() {
    vector<int> & tensorSize = m_pLayer->m_prevLayer->m_tensorSize;

    //The first dimension of the tensor defines the batch size n, and the second dimension defines the number of features maps c.
    int nbDims = tensorSize.size();
    if (1 == ((ConvolutionBasicLayer*) m_pLayer)->m_numInputFeatures){
        nbDims +=2;
    } else{
        nbDims +=1;
    }

    int* dimA = new int[nbDims];
    dimA[0] = 1;
    if (1 == ((ConvolutionBasicLayer*) m_pLayer)->m_numInputFeatures){
        dimA[1] = 1;
        for (int i=2; i< nbDims; ++i){
            dimA[i]  = tensorSize[i-2];
        }
    } else{
        for (int i=1; i< nbDims; ++i){
            dimA[i]  = tensorSize[i-1];
        }
    }

    int* strideA = new int [nbDims];  //span in each dimension. It is a different concept with filter-stride in convolution.
    dimA2SpanA(dimA, nbDims, strideA);

    checkCUDNN(cudnnSetTensorNdDescriptor(m_xDescriptor, CUDNN_DATA_FLOAT, nbDims, dimA, strideA));

    //cout<<"In "<<m_pLayer->m_name<<": ";
    //cout<<"xDescriptor: "<<array2Str(dimA, nbDims)<<endl;

    delete[] dimA;
    delete[] strideA;
}


void CudnnBasicConvolution::setConvDescriptor() {
    // That same convolution descriptor can be reused in the backward path provided it corresponds to the same layer.
    const int  arrayLength = ((ConvolutionBasicLayer*) m_pLayer)->m_filterSize.size();  // this arrayLength does not include n, c.
    int* padA = new int [arrayLength];
    int* filterStrideA  = new int [arrayLength];
    int* dilationA = new int[arrayLength];
    for (int i=0; i< arrayLength; ++i){
        padA[i] = 0;
        filterStrideA[i] = ((ConvolutionBasicLayer*) m_pLayer)->m_stride[i];
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
    const int oneFilterSize = length(((ConvolutionBasicLayer*) m_pLayer)->m_feature_filterSize)*sizeof(float);
    cudaMalloc(&d_pW, oneFilterSize*((ConvolutionBasicLayer*) m_pLayer)->m_numFilters);
    for (int i=0; i< ((ConvolutionBasicLayer*) m_pLayer)->m_numFilters; ++i){
        cudaMemcpy(d_pW+i*oneFilterSize, ((ConvolutionBasicLayer*) m_pLayer)->m_pW[i]->getData(), oneFilterSize, cudaMemcpyHostToDevice);
    }
}

//dW can accumulate
void CudnnBasicConvolution::allocateDevicedW() {
    const int oneFilterSize = length(((ConvolutionBasicLayer*) m_pLayer)->m_feature_filterSize)*sizeof(float);
    cudaMalloc(&d_pdW, oneFilterSize*((ConvolutionBasicLayer*) m_pLayer)->m_numFilters);
    for (int i=0; i< ((ConvolutionBasicLayer*) m_pLayer)->m_numFilters; ++i){
        cudaMemcpy(d_pdW+i*oneFilterSize, ((ConvolutionBasicLayer*) m_pLayer)->m_pdW[i]->getData(), oneFilterSize, cudaMemcpyHostToDevice);
    }
}

void CudnnBasicConvolution::freeDeviceW(){
    if (nullptr != d_pW)  {
        cudaFree(d_pW);
        d_pW= nullptr;
    }
}

void CudnnBasicConvolution::freeDevicedW(){
    if (nullptr != d_pdW)  {
        cudaFree(d_pdW);
        d_pdW= nullptr;
    }
}

void CudnnBasicConvolution::freeWorkSpace(){
    if (nullptr != d_pWorkspace)  {
        cudaFree(d_pWorkspace);
        d_pWorkspace= nullptr;
    }
}


