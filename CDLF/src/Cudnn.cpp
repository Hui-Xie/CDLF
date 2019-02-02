
#include <Cudnn.h>

#include "Cudnn.h"

Cudnn::Cudnn(Layer* pLayer){
    m_pLayer = pLayer;
    d_pX = nullptr;
    d_pY = nullptr;
    d_pdX = nullptr;
    d_pdY = nullptr;
    checkCUDNN(cudnnCreate(&m_cudnnContext));
    checkCUDNN(cudnnCreateTensorDescriptor(&m_xDescriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&m_yDescriptor));

}

Cudnn::~Cudnn(){
   freeDeviceX();
   freeDeviceY();
   freeDevicedX();
   freeDevicedY();
   cudnnDestroyTensorDescriptor(m_xDescriptor);
   cudnnDestroyTensorDescriptor(m_yDescriptor);
   cudnnDestroy(m_cudnnContext);
}

void Cudnn::setXDescriptor() {
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

    cout<<"In "<<m_pLayer->m_name<<": ";
    cout<<"xDescriptor: "<<array2Str(dimA, nbDims)<<endl;

    delete[] dimA;
    delete[] strideA;
}



void Cudnn::allocateDeviceX() {
    freeDeviceX();

    const int xSize = length(m_pLayer->m_prevLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pX, xSize);
    cudaMemcpy(d_pX, m_pLayer->m_prevLayer->m_pYTensor->getData(), xSize, cudaMemcpyHostToDevice);
}

void Cudnn::allocateDeviceY(bool copyComputedY) {
    freeDeviceY();

    const int ySize = length(m_pLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pY, ySize);
    if (copyComputedY){
        cudaMemcpy(d_pY, m_pLayer->m_pYTensor->getData(), ySize, cudaMemcpyHostToDevice);
    }
    else{
       cudaMemset(d_pY, 0, ySize);
    }
}


void Cudnn::allocateDevicedX() {
    freeDevicedX();

    const int xSize = length(m_pLayer->m_prevLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pdX, xSize);
    cudaMemset(d_pdX, 0, xSize);
}

void Cudnn::allocateDevicedY() {
    freeDevicedY();

    const int ySize = length(m_pLayer->m_tensorSize)*sizeof(float);
    cudaMalloc(&d_pdY, ySize);
    cudaMemcpy(d_pdY, m_pLayer->m_pdYTensor->getData(), ySize, cudaMemcpyHostToDevice);
}

void Cudnn::freeDeviceX() {
    if (nullptr != d_pX) {
        cudaFree(d_pX);
        d_pX = nullptr;
    }
}

void Cudnn::freeDeviceY() {
    if (nullptr != d_pY) {
        cudaFree(d_pY);
        d_pY = nullptr;
    }
}

void Cudnn::freeDevicedX() {
    if (nullptr != d_pdX) {
        cudaFree(d_pdX);
        d_pdX = nullptr;
    }
}

void Cudnn::freeDevicedY() {
    if (nullptr != d_pdY) {
        cudaFree(d_pdY);
        d_pdY = nullptr;
    }
}
















