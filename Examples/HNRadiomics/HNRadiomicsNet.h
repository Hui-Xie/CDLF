//
// Created by Hui Xie on 12/26/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_HN_RADIOMICS_H
#define CDLF_HN_RADIOMICS_H

#include "CDLF.h"

class HNRadiomicsNet : public FeedForwardNet {
public:
    HNRadiomicsNet(const string& name, const string& saveDir);
    ~HNRadiomicsNet();

    virtual void build();
    virtual void train();
    virtual float test();
};


#endif //CDLF_HN_RADIOMICS_H
