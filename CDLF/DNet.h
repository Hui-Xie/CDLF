//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_DNET_H
#define CDLF_FRAMEWORK_DNET_H

#include "FeedForwardNet.h"
#include "MergerLayer.h"


class DNet : public FeedForwardNet {
public:
    DNet(const string& name);
    ~DNet();

    Layer* m_pGTLayer;
    Layer* m_pGxLayer;
    Layer* m_pInputXLayer;
    MergerLayer* m_pMerger;

};

#endif //CDLF_FRAMEWORK_DNET_H
