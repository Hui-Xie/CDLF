//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_SEGMENTDNET_H
#define CDLF_FRAMEWORK_SEGMENTDNET_H

#include "DNet.h"


/* Discriminate Network
 * the derived class of DNet is responsible to assign m_pGxLayer, m_pGTLayer,  m_pMerger, and m_pInputXLayer in its build method
 *
 * */

class SegmentDNet : public DNet {
public:
    SegmentDNet(const string& name);
    ~SegmentDNet();

    virtual void build();
    virtual void train();
    virtual float test();

};


#endif //CDLF_FRAMEWORK_SEGMENTDNET_H
