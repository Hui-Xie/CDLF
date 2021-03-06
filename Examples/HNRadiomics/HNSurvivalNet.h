//
// Created by Hui Xie on 02/16/19.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#ifndef HN_SURVIVALNET_H
#define HN_SURVIVALNET_H

#include "CDLF.h"
#include "HNDataManager.h"
#include "HNClinicalDataMgr.h"


class HNSurvivalNet : public FeedForwardNet {
public:
    HNSurvivalNet(const string &netDir);
    ~HNSurvivalNet();

    virtual void build();
    virtual void train();
    virtual float test();

    HNDataManager* m_pDataMgr;
    HNClinicalDataMgr* m_pClinicalDataMgr;

    bool m_printPredict;

    void setInput(const string& filename, const vector<int>& center = vector<int>());
    void setGroundtruth(const string& filename, const bool printPredict = false);

    void printPredict();

    void test(const bool printResult);

};




#endif //HN_SURVIVALNET_H
