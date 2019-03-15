//
// Created by Hui Xie on 10/26/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.
//

#include "CPUAttr.h"
#include <iostream>
#include <thread>

using namespace std;

int CPUAttr::m_numCPUCore = 0;


CPUAttr::CPUAttr() {

}

CPUAttr::~CPUAttr() {

}

void CPUAttr::getCPUAttr(){
    m_numCPUCore =  std::thread::hardware_concurrency();
    cout<<"This host has "<<m_numCPUCore << " CPU cores"<<endl;
}