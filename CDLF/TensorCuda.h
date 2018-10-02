//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_TENSORDEVICE_H
#define CDLF_FRAMEWORK_TENSORDEVICE_H

//cuda is a bridge between Tensor and Device

void cudaZeroInitialize(float* m_data, const long N);

#endif //CDLF_FRAMEWORK_TENSORDEVICE_H
