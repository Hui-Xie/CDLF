//
// Created by hxie1 on 8/29/18.
//

#ifndef CDLF_FRAMEWORK_DAGNET_H
#define CDLF_FRAMEWORK_DAGNET_H

#include "CDLF.h"

class DAGNet : public Net {
public:
    DAGNet();
    ~DAGNet();

    virtual void build();
    virtual void train();
    virtual float test();

};

#endif //CDLF_FRAMEWORK_DAGNET_H
