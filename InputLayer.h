//
// Created by Sheen156 on 6/6/2018.
//

#ifndef RL_NONCONVEX_INPUTLAYER_H
#define RL_NONCONVEX_INPUTLAYER_H

#include "Layer.h"



class InputLayer : public Layer{
public:
    InputLayer(const int id, const string name,long width);
    ~InputLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method);
};


#endif //RL_NONCONVEX_INPUTLAYER_H
