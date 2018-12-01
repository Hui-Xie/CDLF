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

    virtual void quicklySwitchTrainG_D();
    virtual void trainG();
    virtual void trainD();
    virtual float testG(bool outputFile);


    void setDataMgr(DataManager* pDataMgr);
    void setStubNet(StubNetForD* pStubNet);

    void pretrainD();

    StubNetForD* m_pStubNet;
    DataManager* m_pDataMgr;

private:
    void setOneHotLabel(const bool bTrainSet, const int numLabels, const long indexImage, LossLayer* lossLayer, InputLayer* inputLayer);


};


#endif //CDLF_FRAMEWORK_3DSEGMENTATIONNET_H
