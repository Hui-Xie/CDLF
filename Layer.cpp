//
// Created by Sheen156 on 6/5/2018.
//

#include "Layer.h"

Layer::Layer(long width){
    m_name = "";
    m_type = "";
    m_preLayerPointer = nullptr;
    m_width = width;
}


Layer::~Layer(){

}