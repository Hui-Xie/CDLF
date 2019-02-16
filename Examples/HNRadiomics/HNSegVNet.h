//
// Created by Hui Xie on 12/26/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef HN_SEGMENTTATIONVNET_H
#define HN_SEGMENTTATIONVNET_H

#include "CDLF.h"
#include "HNDataManager.h"

class HNSegVNet : public FeedForwardNet {
public:
    HNSegVNet(const string &name, const string &netDir);
    ~HNSegVNet();

    virtual void build();
    virtual void train();
    virtual float test();

    HNDataManager* m_pDataMgr;

    float test(const string& imageFilePath, const string& labelFilePath, const vector<int>& center);

    float m_TPR; // True positive Rate
    float m_dice;
    bool m_isSoftmaxBeforeLoss;

    //void defineAssemblyLoss();

    void detectSoftmaxBeforeLoss();



    void setInput(const string& filename, const vector<int>& center = vector<int>());
    void setGroundtruth(const string& filename, const vector<int>& center = vector<int>());

};




#endif //HN_SEGMENTTATIONVNET_H
