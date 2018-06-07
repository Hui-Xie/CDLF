//
// Created by Sheen156 on 6/5/2018.
//

#include "Layer.h"

Layer::Layer(long width){
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