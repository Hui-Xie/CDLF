//
// Created by Hui Xie on 8/8/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAMEWORK_CROSSENTROPYLOSS_H
#define CDLF_FRAMEWORK_CROSSENTROPYLOSS_H

#include "LossLayer.h"

/*
 * for preLayer is Sigmoid:
 * L= -(1/N)*\sum (p_i * log(x_i) + (1-p_i) *log(1 -x_i))
 * where p_i is the groundtruth distribution
 *       x_i is the output of previous layer, e.g. sigmoid
 *       N is the total number of elements
 *       where log is natural logarithm
 *
 *
 * for prelayer is Softmax:
 * L = -(1/N*C)*\sum p_i* log(x_i)
 * where p_i is the groundtruth distribution
 *       x_i is the output of previous layer, e.g. softmax
 *       N is the total number of elements, and C is the channel number, or C = X.dims[0]
 *       where log is natural logarithm
 *
 *
 * Cross Entropy in concept is same with  Kullback–Leibler divergence
 * Cross entropy supports N-d tensor
 * p_i is binary, and x_i belongs [0,1]
 * */

class CrossEntropyLoss: public LossLayer {
public:
    CrossEntropyLoss(const int id, const string& name,Layer *prevLayer );
    ~CrossEntropyLoss();

    virtual void  printGroundTruth();
    bool predictSuccessInColVec();

    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);


private:
    virtual float lossCompute();
    virtual void  gradientCompute();

};


#endif //CDLF_FRAMEWORK_CROSSENTROPYLOSS_H
