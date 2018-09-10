//
// Created by hxie1 on 9/10/18.
//

#ifndef CDLF_FRAMEWORK_EXPONENTIALLAYER_H
#define CDLF_FRAMEWORK_EXPONENTIALLAYER_H

#include "Layer.h"

/* y_i = exp(x_i)
 * */

class ExponentialLayer : public Layer {
public:
    ExponentialLayer(const int id, const string& name,Layer* prevLayer);
    ~ExponentialLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);
};



#endif //CDLF_FRAMEWORK_EXPONENTIALLAYER_H
