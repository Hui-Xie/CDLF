//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_ITK_RWRITER_H
#define CDLF_FRAMEWORK_ITK_RWRITER_H

#include "Tensor.h"
#include <string>
#include <iostream>
#include "itkImage.h"
#include "itkImageFileReader.h"

// Manage the reading and writing of ITK medical image
template <typename VoxelType, int Dimension>
class ITK_RWriter {
public:
    ITK_RWriter();
    ~ITK_RWriter();

    using ImageType = itk::Image< VoxelType, Dimension >;

    void readFile(const string & filename, Tensor<float>*& pTensor);
    void writeFile(const Tensor<float>* pTensor, const string & filename);

private:
    typename ImageType::PointType m_origin;
    typename ImageType::SizeType m_imageSize;
    typename ImageType::SpacingType m_spacing;




};

#include "ITK_RWriter.hpp"


#endif //CDLF_FRAMEWORK_ITK_RWRITER_H
