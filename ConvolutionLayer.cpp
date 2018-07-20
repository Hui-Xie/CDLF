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
        m_tensorSize = prevLayer->m_tensorSize; // this is initial, not final size

        int N = filterSize.size();
        m_OneFilterN = 1;
        for (int i=0; i<N;++i){
            m_OneFilterN *= filterSize[i];
        }
        addPreviousLayer(prevLayer);
        constructFiltersAndY();
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


void ConvolutionLayer::constructFiltersAndY(){
    for (int i=0; i<m_numFilters;++i){
        m_pW[i] = new Tensor<float>(m_filterSize);
        m_pdW[i] = new Tensor<float>(m_filterSize);
    }

    //get refined pYTensor size
    const int dim = m_tensorSize.size();
    for (int i =0; i<dim; ++i){
        m_tensorSize[i] = ( m_tensorSize[i] -m_filterSize[i])/m_stride +1;
        // ref formula: http://cs231n.github.io/convolutional-networks/
    }
    m_tensorSize.push_back(m_numFilters);

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
    const int N = m_filterSize.size();
    vector<int> Xc = m_filterSize /2; //X central for each subTensor
    Tensor<float>& X = *m_prevLayers.front()->m_pYTensor;
    if (2 == N) {
        for (int i = 0; i < m_tensorSize[0]; ++i){
            for (int j = 0; j < m_tensorSize[1]; ++j){
                Tensor<float> subX = X.subTensor(Xc,m_filterSize);
                for (int idxF =0; idxF < m_numFilters; ++idxF) {  //indexFilter
                     m_pYTensor->e(i,j, idxF)= subX.conv(*m_pW[idxF]);
                }
                Xc = Xc+ m_stride;
            }
         }
    }
    else if (3 == N) {
        for (int i = 0; i < m_tensorSize[0]; ++i){
            for (int j = 0; j < m_tensorSize[1]; ++j){
                for (int k=0; k < m_tensorSize[2]; ++k){
                    Tensor<float> subX = X.subTensor(Xc,m_filterSize);
                    for (int idxF =0; idxF < m_numFilters; ++idxF) {  //indexFilter
                        m_pYTensor->e(i,j, k,idxF)= subX.conv(*m_pW[idxF]);
                    }
                    Xc = Xc+ m_stride;
                }
              }
        }
    }
    else if (4 == N) {
        for (int i = 0; i < m_tensorSize[0]; ++i){
            for (int j = 0; j < m_tensorSize[1]; ++j){
                for (int k=0; k < m_tensorSize[2]; ++k){
                    for (int l=0; l < m_tensorSize[3]; ++l){
                        Tensor<float> subX = X.subTensor(Xc,m_filterSize);
                        for (int idxF =0; idxF < m_numFilters; ++idxF) {  //indexFilter
                            m_pYTensor->e(i,j, k,l,idxF)= subX.conv(*m_pW[idxF]);
                        }
                        Xc = Xc+ m_stride;
                    }
                   }
            }
        }
    }

    else{
        cout<<"Error: dimension does not support in convolution forward."<<endl;
    }
}

void ConvolutionLayer::backward(){

}

void ConvolutionLayer::updateParameters(const float lr, const string& method){

}