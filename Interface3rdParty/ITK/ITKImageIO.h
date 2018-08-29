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


//Manage the reading and writing of ITK medical images

template <typename VoxelType, int Dimension>
class ITKImageIO {
public:
    ITKImageIO();
    ~ITKImageIO();

    using ImageType = itk::Image< VoxelType, Dimension >;

    void readFile(const string & filename, Tensor<float>*& pTensor);
    void writeFileWithSameInputDim(const Tensor<float>* pTensor, const vector<long>& offset, const string & filename);
    void writeFileWithLessInputDim(const Tensor<float>* pTensor, const vector<long>& offset, const string & filename);

private:
    typename ImageType::PointType m_origin;
    typename ImageType::SizeType m_imageSize;
    typename ImageType::SpacingType m_spacing;
    typename ImageType::DirectionType m_direction;
    int m_dim;
};

#include "ITKImageIO.hpp"


#endif //CDLF_FRAMEWORK_ITK_RWRITER_H
