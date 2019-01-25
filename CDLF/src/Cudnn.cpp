
#include <Cudnn.h>

#include "Cudnn.h"

Cudnn::Cudnn(Layer* pLayer){
    m_pLayer = pLayer;
    checkCUDNN(cudnnCreate(&m_cudnnContext));
    checkCUDNN(cudnnCreateTensorDescriptor(&m_xDescriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&m_yDescriptor));

    setXDescriptor();
}

Cudnn::~Cudnn(){
   cudnnDestroyTensorDescriptor(m_xDescriptor);
   cudnnDestroyTensorDescriptor(m_yDescriptor);
   cudnnDestroy(m_cudnnContext);
}

void Cudnn::getDimsArrayFromTensorSize(const vector<int> tensorSize, int& dim, int*& array) {
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

void Cudnn::getDimsArrayFromFilterSize(const vector<int> filterSize, const int numFilters, int &dim, int *&array) {
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

void Cudnn::setXDescriptor() {
    int n=1, c=1, h=1, w =1;
    vector<int> & tensorSize = m_pLayer->m_prevLayer->m_tensorSize;
    if (2 == tensorSize.size()){
        h = tensorSize[0];
        w = tensorSize[1];
    }
    else if (3 ==tensorSize.size()){
        c = tensorSize[0];
        h = tensorSize[1];
        w = tensorSize[2];
    }
    else{
        cout<<"Error: cudnn can not support 4D or above input Tensor in layer "<<m_pLayer->m_name<<endl;
        std::exit(EXIT_FAILURE);
    }
    checkCUDNN(cudnnSetTensor4dDescriptor(m_xDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
}


