//
// Created by Sheen156 on 6/5/2018.
//

#include "Net.h"

Net::Net(){
   m_layers.clear();
}
Net::~Net(){
   while (0 != m_layers.size()){
      Layer* pLayer = m_layers.back();
      m_layers.pop_back();
      delete pLayer;
   }
}

void Net::forwardPropagate(){

}
void Net::backwardPropagate(){

}

void Net::addLayer(Layer* layer){

}

void Net::sgd(const float lr){

}
