//
// Created by Hui Xie on 8/29/18.
//

#ifndef CDLF_FRAMEWORK_DAGNET_H
#define CDLF_FRAMEWORK_DAGNET_H

#include "CDLF.h"

class DAGNet : public FeedForwardNet {
public:
    DAGNet(const string& name);
    ~DAGNet();

    virtual void build();
    void buildSimple();
    virtual void train();
    virtual float test();

};

#endif //CDLF_FRAMEWORK_DAGNET_H
