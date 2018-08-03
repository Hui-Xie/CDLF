//
// Created by Sheen156 on 8/3/2018.
//

#ifndef RL_NONCONVEX_IDENTITYLAYER_H
#define RL_NONCONVEX_IDENTITYLAYER_H
#include "Layer.h"

/* Identity Layer same with  a residual edge
 * Identity's ID should be between the IDs of its previous layer and its next layer
 * */

class IdentityLayer : public Layer {
public:
    IdentityLayer(const int id, const string& name,Layer* prevLayer, Layer* nextLayer);
    ~IdentityLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);
};

#endif //RL_NONCONVEX_IDENTITYLAYER_H
