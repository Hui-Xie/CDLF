//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef CDLF_FRAMEWORK_3DSEGMENTATIONNET_H
#define CDLF_FRAMEWORK_3DSEGMENTATIONNET_H

#include "CDLF.h"
#include "DataManager.h"
#include "StubNetForD.h"


class Segmentation3DNet: public GAN{
public:
    Segmentation3DNet(const string& name, GNet* pGNet, DNet* pDNet);
    ~Segmentation3DNet();

    Tensor<float> constructGroundTruth(Tensor<unsigned char> *pLabels, const long index);

    virtual void trainG(const int N);
    virtual void trainD(const int N);
    virtual float testG();


    void setDataMgr(DataManager* pDataMgr);
    void setStubNet(StubNetForD* pStubNet);

    void pretrainD();

    StubNetForD* m_pStubNet;
    DataManager* m_pDataMgr;


};


#endif //CDLF_FRAMEWORK_3DSEGMENTATIONNET_H
