//
// Created by Hui Xie on 6/7/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#include "statisTool.h"
#include "stats.hpp"
//#define STATS_USE_BLAZE
#include <chrono>
#include <random>
#include <cmath>


// these functions can not GPU parallel.

void generateGaussian(Tensor<float>* yTensor,const float mu, const float sigma ){
    int N = yTensor->getLength();
    unsigned randSeed = std::chrono::system_clock::now().time_since_epoch().count() + rand()%5000;
    std::mt19937_64 randEngine(randSeed);
    for(int i=0; i<N; ++i){
       yTensor->e(i) =  stats::rnorm(mu,sigma,randEngine);
    }
}

// reference: http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
/*
 * Assume:  E(X) = 0; E(W) = 0
 * in the case A : Y = WX  and  W has size of {m*k}, Y has size of {m*n}, X has size of {k*n}
 *        case B:  Y = XW  and  W has size of {k*n}, Y has size of {m*n}, X has size of {m*k}
 *  in    case A and B:  Var(Y) = k * Var(W) *Var(X)
 *  therefore, Var(W) = 1/k;
 *  consider backpropagation: Var(W) = 2/(m+k) or Var(W) = 2/(k+n)
 *  consider ReLU killing half variance: Var(W) = 4/(dim0+dim1)
 *
 * */


void xavierInitialize(Tensor<float>* pW){
    int nRow = pW->getDims()[0];
    int nCol = pW->getDims()[1];
    const float mu =0;
    const float sigma = 4.0/(nCol+nRow); //the variance of output y with consideration of ReLU
    Tensor<float>& W =*pW;
    unsigned randSeed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 randEngine(randSeed);
    const int N = pW->getLength();
    for (int i=0; i<N; ++i){
         W.e(i) = stats::rnorm(mu,sigma,randEngine);
    }
}
