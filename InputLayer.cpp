//
// Created by Sheen156 on 6/6/2018.
//

#include "InputLayer.h"
#include "stats.hpp"
#define STATS_USE_BLAZE
#include <chrono>
#include <random>


InputLayer::InputLayer(const long width): Layer(width){
    m_type = "InputLayer";
    m_pYVector = new DynamicVector<float>(m_width);
    m_pdYVector = new DynamicVector<float>(m_width);
}

InputLayer::~InputLayer(){
    if (nullptr != m_pYVector) delete m_pYVector;
    if (nullptr != m_pdYVector) delete m_pdYVector;
}

void InputLayer::initialize(const string& initialMethod){
    // Gaussian random initialize
    if ("Gaussian" != initialMethod)  return;
    unsigned randSeed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 randEngine(randSeed);
    *m_pYVector =  stats::rnorm(0,1,randEngine);




}