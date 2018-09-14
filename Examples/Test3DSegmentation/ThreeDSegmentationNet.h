//
// Created by Hui Xie on 9/14/18.
// Copyrigh (c) 2018 Hui Xie. All rights reserved.

#ifndef CDLF_FRAMEWORK_3DSEGMENTATIONNET_H
#define CDLF_FRAMEWORK_3DSEGMENTATIONNET_H

#include "CDLF.h"


class ThreeDSegmentationNet: public Net{
public:
    ThreeDSegmentationNet();
    ~ThreeDSegmentationNet();

    Tensor<float> constructGroundTruth(Tensor<unsigned char> *pLabels, const long index);

    virtual void build();
    virtual void train();
    virtual float test();

};


#endif //CDLF_FRAMEWORK_3DSEGMENTATIONNET_H
