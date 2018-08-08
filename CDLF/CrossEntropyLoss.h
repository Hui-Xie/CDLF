//
// Created by Sheen156 on 8/8/2018.
//

#ifndef CDLF_FRAMEWORK_CROSSENTROPYLOSS_H
#define CDLF_FRAMEWORK_CROSSENTROPYLOSS_H

#include "LossLayer.h"

/* L= -\sum p_i * log(x_i)
 * where p_i is the groundtruth distribution
 *       x_i is the output of previous layer, e.g. softmax;
 *       where log is natural logarithm
 * Cross Entropy in concept is same with  Kullback–Leibler divergence
 * */

class CrossEntropyLoss: public LossLayer {
public:
    CrossEntropyLoss(const int id, const string& name);
    ~CrossEntropyLoss();

private:
    virtual float lossCompute();
    virtual void  gradientCompute();
    virtual void  printGroundTruth();
};


#endif //CDLF_FRAMEWORK_CROSSENTROPYLOSS_H
