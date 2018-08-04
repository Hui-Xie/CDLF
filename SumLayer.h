//
// Created by Hui Xie on 8/4/2018.
//

#ifndef RL_NONCONVEX_SUMLAYER_H
#define RL_NONCONVEX_SUMLAYER_H

#include "Layer.h"

/* Y = \sum X_i
 * dL/dx_i = dL/dY * dY/dX_i = dL/dY;
 * in the BranchLayer, the dL/dY should be accumulate from the following layers of SumLayer
 * */

class SumLayer : public Layer {
public:
    SumLayer(const int id, const string& name, const vector<int>& tensorSize);
    ~SumLayer();

    list<Layer*> m_prevLayers;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);

    virtual void addPreviousLayer(Layer* prevLayer);
};


#endif //RL_NONCONVEX_SUMLAYER_H
