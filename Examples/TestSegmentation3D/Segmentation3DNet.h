//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAMEWORK_3DSEGMENTATIONNET_H
#define CDLF_FRAMEWORK_3DSEGMENTATIONNET_H

#include "CDLF.h"
#include "Seg3DDataManager.h"
#include "StubNetForD.h"


class Segmentation3DNet: public GAN{
public:
    Segmentation3DNet(const string& name, GNet* pGNet, DNet* pDNet);
    ~Segmentation3DNet();

    virtual void quicklySwitchTrainG_D();
    virtual void trainG();
    virtual void trainD();
    virtual float testG(bool outputFile);


    void setDataMgr(Seg3DDataManager* pDataMgr);
    void setStubNet(StubNetForD* pStubNet);

    void pretrainD();

    StubNetForD* m_pStubNet;
    Seg3DDataManager* m_pDataMgr;

private:
    void setOneHotLabel(const bool bTrainSet, const int numLabels, const int indexImage, LossLayer* lossLayer, InputLayer* inputLayer);


};


#endif //CDLF_FRAMEWORK_3DSEGMENTATIONNET_H
