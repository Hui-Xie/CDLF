//
// Created by Hui Xie on 9/15/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "GANet.h"


GANet::GANet(){

}

GANet::~GANet(){

}

void GANet::forwardG(){

}

void GANet::forwardD(){

}

void GANet::backwardG(){

}

void GANet::backwardD(){

}

void GANet::sgdG(){

}

void GANet::sgdD(){

}

bool GANet::checkLayerAttribute(){
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        if (0 == iter->second->getAttribute()){
            cout<<"Error: incorrect layer attribute: ID="<<iter->second->m_id<<"; name="<<iter->second->m_name<<endl;
            return false;
        }
    }
    return true;
}
