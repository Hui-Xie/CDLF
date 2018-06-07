//
// Created by Sheen156 on 6/6/2018.
//

#ifndef RL_NONCONVEX_INPUTLAYER_H
#define RL_NONCONVEX_INPUTLAYER_H

#include "Layer.h"


class InputLayer : public Layer{
public:
    InputLayer(long width);
    ~InputLayer();

    virtual  void initialize(const string& initialMethod);
};


#endif //RL_NONCONVEX_INPUTLAYER_H
