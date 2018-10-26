//
// Created by Hui Xie on 10/26/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_CPUATTR_H
#define CDLF_FRAMEWORK_CPUATTR_H


class CPUAttr{
public:
    CPUAttr();
    ~CPUAttr();

    static  int m_numCPUCore;

    void getCPUAttr();
};


#endif //CDLF_FRAMEWORK_CPUATTR_H
