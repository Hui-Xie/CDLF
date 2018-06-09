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
    m_pYVector = nullptr;
    m_pdYVector = nullptr;
    m_width = width;
}


Layer::~Layer(){

}

void Layer::printY(){
    cout<<"LayerName: "<<m_name <<"; LayerWidth: "<<m_width<<";  Y: ";
    printVector(m_pYVector);
}

void Layer::printDY(){
    cout<<"LayerName: "<<m_name <<"; LayerWidth: "<<m_width<<"; dY: ";
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