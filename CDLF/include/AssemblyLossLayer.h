//
// Created by Hui Xie on 01/29/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_ASSEMBLYLOSS_H
#define CDLF_FRAMEWORK_ASSEMBLYLOSS_H

#include "LossLayer.h"
#include <list>

/*  Assembly Loss: sum of various loss functions
 *
 * */

class AssemblyLossLayer  : public LossLayer {
public:
    AssemblyLossLayer(const int id, const string& name,  Layer *prevLayer);
    ~AssemblyLossLayer();

    list<LossLayer*> m_lossList;

    void addLoss(LossLayer* lossLayer);
    template<typename T> void setGroundTruth( const Tensor<T>& groundTruth);

    virtual  void printStruct();

private:
    virtual float lossCompute();
    virtual void  gradientCompute();
};


template<typename T>
void AssemblyLossLayer::setGroundTruth( const Tensor<T>& groundTruth){
    list<LossLayer*>::iterator it = m_lossList.begin();
    while( it != m_lossList.end()){
        (*it)->setGroundTruth<T>(groundTruth);
        ++it;
    }
}

#endif //CDLF_FRAMEWORK_ASSEMBLYLOSS_H

