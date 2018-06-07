//
// Created by Sheen156 on 6/5/2018.
//

#ifndef RL_NONCONVEX_LAYER_H
#define RL_NONCONVEX_LAYER_H

#include <string>
#include  <blaze/Math.h>
using namespace blaze;
using namespace std;

class Layer {
public:
    Layer(long width);
    ~Layer();

    string m_name;
    string m_type;
    Layer* m_preLayerPointer;
    long m_width;
    DynamicVector<float>*  m_pYVector;

    virtual  void forward()=0;
    virtual  void backward()=0;
    virtual  void initialize(const string& initialMethod)=0;

};


#endif //RL_NONCONVEX_LAYER_H
