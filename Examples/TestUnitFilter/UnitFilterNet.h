//
// Created by Hui Xie on 8/16/2018.
// Copyrigh (c) 2018 Hui Xie. All rights reserved.


#ifndef CDLF_FRAMEWORK_CONVEXNET_H
#define CDLF_FRAMEWORK_CONVEXNET_H

#include "CDLF.h"

class UnitFilterNet : public Net {
public:
    UnitFilterNet();
    ~UnitFilterNet();

    virtual void build();
    virtual void train();
    virtual float test();

};


#endif //CDLF_FRAMEWORK_CONVEXNET_H
