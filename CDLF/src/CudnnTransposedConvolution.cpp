
#include <CudnnTransposedConvolution.h>


CudnnTransposedConvolution::CudnnTransposedConvolution(TransposedConvolutionLayer *pLayer, const vector<int> &filterSize, const int numFilters,
                                   const int stride): CudnnBasicConvolution(pLayer, filterSize, numFilters, stride)
{
  //null
}


CudnnTransposedConvolution::~CudnnTransposedConvolution(){
  //null
}

void CudnnTransposedConvolution::forward() {
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

    if (nullptr != d_pWorkspace){
        cudaFree(d_pWorkspace);
        d_pWorkspace = nullptr;
    }
}

void CudnnTransposedConvolution::backward(bool computeW, bool computeX) {
    allocateDevicedY();

    if (computeW){
        allocateDeviceX();
        allocateDevicedW();
        setBackWardFilterAlg();
        checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(m_cudnnContext, m_xDescriptor, m_yDescriptor, m_convDescriptor, m_wDescriptor,
                                                                  m_bwdFilterAlg, &m_workspaceSize));
        cudaMalloc(&d_pWorkspace, m_workspaceSize);

        float alpha = 1;
        float beta = 1; //for dw to accumulate
        checkCUDNN(cudnnConvolutionBackwardFilter(m_cudnnContext, &alpha,
                                                  m_xDescriptor, d_pX,
                                                  m_yDescriptor, d_pdY,
                                                  m_convDescriptor, m_bwdFilterAlg,
                                                  d_pWorkspace, m_workspaceSize,
                                                  &beta,
                                                  m_wDescriptor, d_pdW));

        const int wSize = length(m_filterSize);
        for (int i=0; i< m_numFilters; ++i){
            cudaMemcpy(((TransposedConvolutionLayer*)m_pLayer)->m_pdW[i]->getData(), d_pdW+i*wSize, wSize* sizeof(float), cudaMemcpyDeviceToHost);
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
        const int xSize = length(m_pLayer->m_prevLayer->m_tensorSize)*sizeof(float);
        cudaMemcpy(m_pLayer->m_prevLayer->m_pdYTensor->getData(), d_pdX, xSize, cudaMemcpyDeviceToHost);

        if (nullptr != d_pWorkspace){
            cudaFree(d_pWorkspace);
            d_pWorkspace = nullptr;
        }
    }
}

void CudnnTransposedConvolution::setYDescriptor() {
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

void CudnnTransposedConvolution::setForwardAlg() {
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(m_cudnnContext, m_xDescriptor, m_wDescriptor, m_convDescriptor, m_yDescriptor,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &m_fwdAlg));
}

void CudnnTransposedConvolution::setBackwardDataAlg(){
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(m_cudnnContext, m_wDescriptor, m_yDescriptor, m_convDescriptor, m_xDescriptor,
                                                        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &m_bwdDataAlg));
}
void CudnnTransposedConvolution::setBackWardFilterAlg(){
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(m_cudnnContext, m_xDescriptor, m_yDescriptor, m_convDescriptor, m_wDescriptor,
                                                          CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &m_bwdFilterAlg));
}