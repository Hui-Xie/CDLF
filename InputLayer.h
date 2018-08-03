//
// Created by Sheen156 on 6/6/2018.
//

#ifndef RL_NONCONVEX_INPUTLAYER_H
#define RL_NONCONVEX_INPUTLAYER_H

#include "Layer.h"



class InputLayer : public Layer{
public:
    InputLayer(const int id, const string& name,const vector<int>& tensorSize);
    ~InputLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);
};


#endif //RL_NONCONVEX_INPUTLAYER_H
