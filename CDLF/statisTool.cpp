//
// Created by Hui Xie on 6/7/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "statisTool.h"
#include "stats.hpp"
//#define STATS_USE_BLAZE
#include <chrono>
#include <random>
#include <cmath>


// these functions can not GPU parallel.

void generateGaussian(Tensor<float>* yTensor,const float mu, const float sigma ){
    long N = yTensor->getLength();
    unsigned randSeed = std::chrono::system_clock::now().time_since_epoch().count() + rand()%5000;
    std::mt19937_64 randEngine(randSeed);
    for(long i=0; i<N; ++i){
       yTensor->e(i) =  stats::rnorm(mu,sigma,randEngine);
    }
}

void xavierInitialize(Tensor<float>* pW){
    long nRow = pW->getDims()[0];
    long nCol = pW->getDims()[1];
    const float mu =0;
    const float sigma = 2.0/nCol; //the variance of output y with consideration of ReLU
    Tensor<float>& W =*pW;
    unsigned randSeed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 randEngine(randSeed);
    for (long i=0; i<nRow; ++i){
        for (long j=0; j<nCol; ++j){
            W.e({i,j}) = stats::rnorm(mu,sigma,randEngine);
        }
    }
}
