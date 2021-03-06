#include <CudnnConvolution.h>


CudnnConvolution::CudnnConvolution(ConvolutionLayer *pLayer): CudnnBasicConvolution(pLayer)
{
    setDescriptors();
}


CudnnConvolution::~CudnnConvolution(){
    //null
}

//Y = W*X+b
void CudnnConvolution::forward() {
    allocateDeviceY();
    allocateDeviceX();
    allocateDeviceW();
    allocateDeviceB();
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
                            m_yDescriptor, d_pY));  // Y = W*X
    beta = 1;
    checkCUDNN(cudnnAddTensor(m_cudnnContext, &alpha,
                              m_bDescriptor, d_pB,
                              &beta,
                              m_yDescriptor, d_pY)); // Y += b

    const size_t ySize = length(m_pLayer->m_tensorSize);
    cudaMemcpy(m_pLayer->m_pYTensor->getData(), d_pY, ySize*sizeof(float), cudaMemcpyDeviceToHost);

    freeWorkSpace();
    freeDeviceY();
    freeDeviceX();
    freeDeviceW();
    freeDeviceB();
}

void CudnnConvolution::backward(bool computeW, bool computeX) {
    allocateDevicedY();

    if (computeW){
        allocateDeviceX();
        allocateDevicedW();
        allocateDevicedB();
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

        const size_t wSize = length(((ConvolutionBasicLayer*) m_pLayer)->m_feature_filterSize);
        const int numFilters = ((ConvolutionBasicLayer*) m_pLayer)->m_numFilters;
        for (int i=0; i< numFilters; ++i){
            cudaMemcpy(((ConvolutionBasicLayer*) m_pLayer)->m_pdW[i]->getData(), d_pdW+i*wSize, wSize* sizeof(float), cudaMemcpyDeviceToHost);
        }
        freeDeviceX();
        freeDevicedW();

        checkCUDNN(cudnnConvolutionBackwardBias(m_cudnnContext, &alpha,
                                                m_yDescriptor, d_pdY,
                                                &beta,
                                                m_bDescriptor, d_pdB));
        cudaMemcpy(((ConvolutionBasicLayer*) m_pLayer)->m_pdB->getData(), d_pdB,  numFilters*sizeof(float), cudaMemcpyDeviceToHost);

        freeWorkSpace();
        freeDevicedB();
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
        const size_t xSize = length(m_pLayer->m_prevLayer->m_tensorSize);
        cudaMemcpy(m_pLayer->m_prevLayer->m_pdYTensor->getData(), d_pdX, xSize*sizeof(float), cudaMemcpyDeviceToHost);

        freeWorkSpace();
        freeDevicedX();
        freeDeviceW();
    }
    freeDevicedY();
}

void CudnnConvolution::setWDescriptor() {
    //The first dimension of the tensor defines number of output features, and the second dimension defines the number of input features maps.
    const int filterDim = ((ConvolutionBasicLayer*) m_pLayer)->m_feature_filterSize.size();
    const int nbDims = filterDim+1;

    int* filterDimA = new int[nbDims];
    filterDimA[0] = ((ConvolutionBasicLayer*) m_pLayer)->m_numFilters;
    for (int i=1; i< nbDims; ++i){
        filterDimA[i]  = ((ConvolutionBasicLayer*) m_pLayer)->m_feature_filterSize[i-1];
    }

    checkCUDNN(cudnnSetFilterNdDescriptor(m_wDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, nbDims, filterDimA));

    //cout<<"In "<<m_pLayer->m_name<<": ";
    //cout<<"wDescriptor: "<<array2Str(filterDimA, nbDims)<<endl;

    delete[] filterDimA;
}

void CudnnConvolution::setYDescriptor() {
    int nbDims = ((ConvolutionBasicLayer*) m_pLayer)->m_filterSize.size()+2;
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

        //cout<<"In "<<m_pLayer->m_name<<": ";
        //cout<<"yDescriptor: "<<array2Str(tensorOuputDimA, nbDims)<<endl;

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