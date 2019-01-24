
#include <Cudnn.h>

#include "Cudnn.h"

Cudnn::Cudnn(Layer* pLayer, const int stride){
    m_pLayer = pLayer;
    m_stride = stride;
    cudnnCreate(&m_cudnnContext);
    cudnnCreateTensorDescriptor(&m_xDescriptor);
    cudnnCreateTensorDescriptor(&m_yDescriptor);

    setDescriptors();
}

Cudnn::~Cudnn(){
   cudnnDestroyTensorDescriptor(m_xDescriptor);
   cudnnDestroyTensorDescriptor(m_yDescriptor);
   cudnnDestroy(m_cudnnContext);
}

void Cudnn::getDimsArrayFromTensorSize(const vector<int> tensorSize, int*& array, int& dim) {
    const int oldDim = tensorSize.size();
    const int newDim  = (oldDim < 4)? 4: oldDim;
    dim = newDim;
    array = new int[newDim];
    for (int i=0; i<newDim; ++i) {
        array[i] = 1;
    }
    for (int i=0; i<oldDim; ++i){
        array[newDim-oldDim+i]= tensorSize[i];
    }
}

void Cudnn::getDimsArrayFromFilterSize(const vector<int> filterSize, const int numFilters, int *&array, int &dim) {
    const int oldDim = filterSize.size();
    dim = oldDim +1;
    array = new int[dim];
    array[0] = numFilters;
    for (int i=1; i<dim; ++i){
        array[i]= filterSize[i-1];
    }
}



void Cudnn::generateStrideArray(const int stride, const int dim, int *&array) {
   array = new int[dim];
   for (int i =0; i<dim; ++i){
       array[i] = stride;
   }
}

void Cudnn::setDescriptors() {
    // input descriptor
    int dim = 0;
    int* xDimsA = nullptr;
    int* strideArray = nullptr;
    getDimsArrayFromTensorSize(m_pLayer->m_prevLayer->m_tensorSize, xDimsA, dim);
    generateStrideArray(m_stride,dim, strideArray);
    cudnnSetTensorNdDescriptor(m_xDescriptor, CUDNN_DATA_FLOAT, dim, xDimsA,strideArray);

    delete xDimsA;
    delete strideArray;

    // output descriptor
    int* yDimsA = nullptr;
    getDimsArrayFromTensorSize(m_pLayer->m_tensorSize, yDimsA, dim);
    generateStrideArray(m_stride,dim, strideArray);
    cudnnSetTensorNdDescriptor(m_yDescriptor, CUDNN_DATA_FLOAT, dim, yDimsA,strideArray);

    delete yDimsA;
    delete strideArray;
}

