//
// Created by Sheen156 on 6/5/2018.
//

#include "Layer.h"
#include <iostream>
using namespace std;

Layer::Layer(const int id, const string& name, const vector<int>& tensorSize){
    m_id = id;
    m_name = name;
    m_type = "";
    m_prevLayers.clear();
    m_nextLayers.clear();

    m_tensorSize = tensorSize;
    if (0 != m_tensorSize.size()){
        m_pYTensor = new Tensor<float>(m_tensorSize);
        m_pdYTensor = new Tensor<float>(m_tensorSize);
    }
    else{
        m_pYTensor = nullptr;
        m_pdYTensor = nullptr;
    }
}


Layer::~Layer(){
    if (nullptr != m_pYTensor) delete m_pYTensor;
    if (nullptr != m_pdYTensor) delete m_pdYTensor;
}

void Layer::printY(){
    cout<<"LayerType: "<<m_type <<"; LayerTensorSize: "<<vector2String(m_tensorSize)<<"; Y: "<<endl;;
    m_pYTensor->printElements();
}

void Layer::printDY(){
    cout<<"LayerType: "<<m_type <<"; LayerTensorSize: "<<vector2String(m_tensorSize)<<"; dY: "<<endl;
    m_pdYTensor->printElements();
}

/*
void Layer::printVector(Tensor<float>* vector){
    Tensor<float>& Y =   *vector;
    cout<<"( ";
    for (long i=0; i< m_tensorSize; ++i){
        if (i != m_tensorSize-1 ) cout<<Y[i]<<", ";
        else cout<<Y[i];
    }
    cout<<" )"<<endl;
}
*/

void Layer::addPreviousLayer(Layer* preLayer){
    if (nullptr != preLayer){
        m_prevLayers.push_back(preLayer);
        preLayer->m_nextLayers.push_back(this);
    }
}