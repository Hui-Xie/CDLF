//
// Created by Hui Xie on 12/26/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#ifndef HN_SEGMENTTATIONVNET_H
#define HN_SEGMENTTATIONVNET_H

#include "CDLF.h"
#include "HNDataManager.h"

class HNSegVNet : public FeedForwardNet {
public:
    HNSegVNet(const string &netDir);
    ~HNSegVNet();

    virtual void build();
    virtual void train();
    virtual float test();

    HNDataManager* m_pDataMgr;

    float test(const string& imageFilePath, const string& labelFilePath);

    float m_TPR; // True positive Rate
    float m_dice;
    bool m_isSoftmaxBeforeLoss;

    //void defineAssemblyLoss();

    void detectSoftmaxBeforeLoss();


    void setGroundtruth(const string &filename, const vector<float>& radianVec, vector<int>& center, const int translationMaxValue);
    void setInput(const string &filename, const vector<float>& radianVec, const vector<int>& center);


};




#endif //HN_SEGMENTTATIONVNET_H
