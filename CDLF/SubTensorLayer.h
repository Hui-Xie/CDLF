//
// Created by hxie1 on 9/13/18.
//

#ifndef CDLF_FRAMEWORK_SUBTENSORLAYER_H
#define CDLF_FRAMEWORK_SUBTENSORLAYER_H

#include "Layer.h"

/* subTensorLayer extract a subTensor from previous layer.
 *
 * */

class SubTensorLayer : public Layer {
public:

    SubTensorLayer(const int id, const string &name, Layer *prevLayer, const vector<long>& start, const vector<long>& span);

    ~SubTensorLayer();

    virtual void initialize(const string &initialMethod);

    virtual void zeroParaGradient();

    virtual void forward();

    virtual void backward();

    virtual void updateParameters(const float lr, const string &method, const int batchSize = 1);

private:
    vector<long> m_start;
    vector<long> m_span;

};




#endif //CDLF_FRAMEWORK_SUBTENSORLAYER_H
