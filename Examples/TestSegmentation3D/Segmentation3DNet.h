//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef CDLF_FRAMEWORK_3DSEGMENTATIONNET_H
#define CDLF_FRAMEWORK_3DSEGMENTATIONNET_H

#include "CDLF.h"


class Segmentation3DNet: public GANet{
public:
    Segmentation3DNet();
    ~Segmentation3DNet();

    Tensor<float> constructGroundTruth(Tensor<unsigned char> *pLabels, const long index);

    virtual float test();

    virtual void buildG(); //Generative network
    virtual void buildD(); //Discriminative network
    virtual void trainG();
    virtual void trainD();

};


#endif //CDLF_FRAMEWORK_3DSEGMENTATIONNET_H
