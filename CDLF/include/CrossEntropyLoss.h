//
// Created by Hui Xie on 8/8/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef CDLF_FRAMEWORK_CROSSENTROPYLOSS_H
#define CDLF_FRAMEWORK_CROSSENTROPYLOSS_H

#include "LossLayer.h"

/* L= -\sum p_i * log(x_i)
 * where p_i is the groundtruth distribution
 *       x_i is the output of previous layer, e.g. softmax;
 *       where log is natural logarithm
 * Cross Entropy in concept is same with  Kullback–Leibler divergence
 * Cross entropy supports N-d tensor
 * */

class CrossEntropyLoss: public LossLayer {
public:
    CrossEntropyLoss(const int id, const string& name,Layer *prevLayer );
    ~CrossEntropyLoss();

    virtual void  printGroundTruth();
    bool predictSuccessInColVec();
    float diceCoefficient();
    float getTPR();

    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);


private:
    virtual float lossCompute();
    virtual void  gradientCompute();

};


#endif //CDLF_FRAMEWORK_CROSSENTROPYLOSS_H
