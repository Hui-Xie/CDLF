//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_DNET_H
#define CDLF_FRAMEWORK_DNET_H

#include "FeedForwardNet.h"
#include "MergerLayer.h"
#include "CrossEntropyLoss.h"


/* Discriminate Network
 * the derived class of DNet is responsible to assign m_pGxLayer, m_pGTLayer,  m_pMerger, and m_pInputXLayer in its build method
 * alpha is the final out of DNet
 * [0,1]' indicate alpha = true;
 * [1,0]' indicate alpha = false;
 *
 * */

class DNet : public FeedForwardNet {
public:
    DNet(const string& name, const string& saveDir);
    ~DNet();

    InputLayer* m_pGTLayer;
    Layer* m_pGxLayer;
    InputLayer* m_pInputXLayer;
    MergerLayer* m_pMerger;
    CrossEntropyLoss* m_pLossLayer;

    void setAlphaGroundTruth(bool alpha);

    virtual void save();
    virtual void load();

    void saveDNetParameters();
    void loadDNetParameters();

};

#endif //CDLF_FRAMEWORK_DNET_H
