//
// Created by Sheen156 on 7/19/2018.
//

#include "ConvolutionLayer.h"
#include "statisTool.h"


ConvolutionLayer::ConvolutionLayer(const int id, const string& name, const vector<int>& filterSize,
                                   Layer* prevLayer, const int numFilters, const int stride)
: Layer(id,name,{})
{
    if (checkFilterSize(filterSize, prevLayer)){
        m_type = "Convolution";
        m_stride = stride;
        m_filterSize = filterSize;
        m_numFilters = numFilters;

        int N = filterSize.size();
        m_OneFilterN = 1;
        for (int i=0; i<N;++i){
            m_OneFilterN *= filterSize[i];
        }
        addPreviousLayer(prevLayer);
        constructFilterAndY(filterSize, prevLayer, numFilters);
    }
    else{
        cout<<"Error: can not construct Convolution Layer: "<<name<<endl;
    }

}

ConvolutionLayer::~ConvolutionLayer(){
    //delete Filter Space; the Y space  will delete by base class;
    if (nullptr != m_pW){
        delete m_pW;
        m_pW = nullptr;
    }
    if (nullptr != m_pdW){
        delete m_pdW;
        m_pdW = nullptr;
    }

}

bool ConvolutionLayer::checkFilterSize(const vector<int>& filterSize, Layer* prevLayer){
    int dimFilter = filterSize.size();
    int dimX = prevLayer->m_tensorSize.size();
    if (dimFilter == dimX){
        for (int i= 0; i< dimX; ++i){
            if (0 == filterSize[i]%2){
                cout<<"Error: filter Size should be odd."<<endl;
                return false;
             }
        }
        return true;
    }
    else{
        cout<<"Error: dimension of filter should be consistent with the previous Layer."<<endl;
        return false;
    }
}


void ConvolutionLayer::constructFilterAndY(const vector<int>& filterSize, Layer* prevLayer,
                                           const int numFilters, const int stride){
    m_pW = new Tensor<float>(filterSize);
    m_pdW = new Tensor<float>(filterSize);
    const vector<int>& prevTensorSize = prevLayer->m_tensorSize;
    m_tensorSize.clear();
    int dim = prevTensorSize.size();
    for (int i =0; i<dim; ++i){
        int range = prevTensorSize[i]- filterSize[i] +1;
        m_tensorSize.push_back(range);
    }
    if (0 != m_tensorSize.size()){
        m_pYTensor = new Tensor<float>(m_tensorSize);
        m_pdYTensor = new Tensor<float>(m_tensorSize);
    }
    else{
        m_pYTensor = nullptr;
        m_pdYTensor = nullptr;
    }
}


void ConvolutionLayer::initialize(const string& initialMethod){
    generateGaussian(m_pW, 0, sqrt(1.0/m_OneFilterN));
}

// Y = W*X
void ConvolutionLayer::forward(){



}

void ConvolutionLayer::backward(){

}

void ConvolutionLayer::updateParameters(const float lr, const string& method){

}