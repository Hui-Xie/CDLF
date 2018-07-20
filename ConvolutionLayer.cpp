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
        constructFilterAndY(filterSize, prevLayer, numFilters, stride);
    }
    else{
        cout<<"Error: can not construct Convolution Layer: "<<name<<endl;
    }

}

ConvolutionLayer::~ConvolutionLayer(){
    //delete Filter Space; the Y space  will delete by base class;
    for(int i=0; i< m_numFilters;++i){
        if (nullptr != m_pW[i]){
            delete m_pW[i];
            m_pW[i] = nullptr;
        }
        if (nullptr != m_pdW[i]){
            delete m_pdW[i];
            m_pdW[i] = nullptr;
        }
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
    for (int i=0; i<numFilters;++i){
        m_pW[i] = new Tensor<float>(filterSize);
        m_pdW[i] = new Tensor<float>(filterSize);
    }

    //get pYTensor size
    m_tensorSize = prevLayer->m_tensorSize;
    const int dim = m_tensorSize.size();
    for (int i =0; i<dim; ++i){
        m_tensorSize[i] += 1-filterSize[i];
    }
    m_tensorSize.push_back(numFilters);

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
    for(int i=0; i< m_numFilters; ++i){
        generateGaussian(m_pW[i], 0, sqrt(1.0/m_OneFilterN));
    }
}

// Y = W*X
void ConvolutionLayer::forward(){



}

void ConvolutionLayer::backward(){

}

void ConvolutionLayer::updateParameters(const float lr, const string& method){

}