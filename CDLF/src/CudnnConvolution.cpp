
#include <CudnnConvolution.h>

CudnnConvolution::CudnnConvolution(Layer *pLayer, const vector<int> &filterSize, const int numFilters,
                                   const int stride): Cudnn(pLayer, stride)
{
    m_numFilters = numFilters;
    cudnnCreateFilterDescriptor(&m_filterDescriptor);
    cudnnCreateConvolutionDescriptor(&m_convDescriptor);

    getDimsArrayFromFilterSize(filterSize,m_numFilters, m_filterSizeArray, m_filterArrayDim);


}


CudnnConvolution::~CudnnConvolution(){
    cudnnDestroyFilterDescriptor(m_filterDescriptor);
    cudnnDestroyConvolutionDescriptor(m_convDescriptor);
    delete m_filterSizeArray;
}

void CudnnConvolution::setConvDescriptors() {
    cudnnSetFilterNdDescriptor(m_filterDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, m_filterArrayDim, m_filterSizeArray);
    cudnnSetConvolution2dDescriptor(m_convDescriptor, 0,0, m_stride, m_stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
}
