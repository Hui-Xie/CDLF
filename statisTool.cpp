//
// Created by Sheen156 on 6/7/2018.
//

#include "statisTool.h"
#include "stats.hpp"
#define STATS_USE_BLAZE
#include <chrono>
#include <random>

void generateGaussian(DynamicVector<float>* yVector,const float mu, const float sigma ){
    long N = yVector->size();
    DynamicVector<float>& vector = *yVector;
    unsigned randSeed = std::chrono::system_clock::now().time_since_epoch().count() + rand()%1000;
    std::mt19937_64 randEngine(randSeed);
    for(int i=0; i<N; ++i){
       vector[i] =  stats::rnorm(mu,sigma,randEngine);
    }
}

void xavierInitialize(DynamicMatrix<float>* pW){
    long nRow = pW->rows();
    long nCol = pW->columns();
    const float mu =0;
    const float sigma = 2.0/nCol; //the variance of output y with consideration of ReLU
    DynamicMatrix<float>& W =*pW;
    unsigned randSeed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 randEngine(randSeed);
    for (long i=0; i<nRow; ++i){
        for (long j=0; j<nCol; ++j){
            W(i,j) = stats::rnorm(mu,sigma,randEngine);
        }
    }
}

