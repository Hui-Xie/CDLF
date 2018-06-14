//
// Created by Sheen156 on 6/5/2018.
//

#include "Layer.h"
#include <iostream>
using namespace std;

Layer::Layer(const long width){
    m_name = "";
    m_type = "";
    m_prevLayerPointer = nullptr;
    m_nextLayerPointer = nullptr;

    m_width = width;
    if (0 != m_width){
        m_pYVector = new DynamicVector<float>(m_width);
        m_pdYVector = new DynamicVector<float>(m_width);
    }
    else{
        m_pYVector = nullptr;
        m_pdYVector = nullptr;
    }

}


Layer::~Layer(){
    if (nullptr != m_pYVector) delete m_pYVector;
    if (nullptr != m_pdYVector) delete m_pdYVector;
}

void Layer::printY(){
    cout<<"LayerType: "<<m_type <<"; LayerWidth: "<<m_width<<"; Y: ";
    printVector(m_pYVector);
}

void Layer::printDY(){
    cout<<"LayerType: "<<m_type <<"; LayerWidth: "<<m_width<<"; dY: ";
    printVector(m_pdYVector);
}

void Layer::printVector(DynamicVector<float>* vector){
    DynamicVector<float>& Y =   *vector;
    cout<<"( ";
    for (long i=0; i< m_width; ++i){
        if (i != m_width-1 ) cout<<Y[i]<<", ";
        else cout<<Y[i];
    }
    cout<<" )"<<endl;
}

void Layer::setPreviousLayer(Layer* preLayer){
    if (nullptr != preLayer){
        m_prevLayerPointer = preLayer;
        m_prevLayerPointer->m_nextLayerPointer = this;
    }
    else{
        m_prevLayerPointer = nullptr;
        m_nextLayerPointer = nullptr;
    }
}