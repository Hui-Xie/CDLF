
#include <CudnnConvolution.h>


CudnnConvolution::CudnnConvolution(ConvolutionLayer *pLayer, const vector<int> &filterSize, const int numFilters,
                                   const int stride): CudnnBasicConvolution(pLayer, filterSize, numFilters, stride)
{
    //null
}


CudnnConvolution::~CudnnConvolution(){
    //null
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

    if (nullptr != d_pWorkspace){
        cudaFree(d_pWorkspace);
        d_pWorkspace = nullptr;
    }
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
        const int xSize = length(m_pLayer->m_prevLayer->m_tensorSize)*sizeof(float);
        cudaMemcpy(m_pLayer->m_prevLayer->m_pdYTensor->getData(), d_pdX, xSize, cudaMemcpyDeviceToHost);

        if (nullptr != d_pWorkspace){
            cudaFree(d_pWorkspace);
            d_pWorkspace = nullptr;
        }
    }
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
        dimA2SpanA(tensorOuputDimA, nbDims,strideA);

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