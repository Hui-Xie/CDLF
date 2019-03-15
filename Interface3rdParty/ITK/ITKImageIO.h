//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.
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

    void readFile(const string & filename, Tensor<VoxelType>*& pTensor);
    void readLabelFileAndOrigin(const string & labelFilename, Tensor<VoxelType>*& pTensor, typename itk::Image<VoxelType, Dimension>::PointType& labelOrigin) const;

    // the offset below should be ITK image dimension order, instead of Tensor Image dimension order
    template<typename OutputValueType> void writeFileWithSameInputDim(const Tensor<OutputValueType>* pTensor, const vector<int>& offset, const string & filename);
    template<typename OutputValueType> void writeFileWithLessInputDim(const Tensor<OutputValueType>* pTensor, const vector<int>& offset, const string & filename);

    //make label file has same volume with original intensity file
    void extendLabelFileVolume(const string & labelFilename, Tensor<VoxelType>*& pTensor);
    vector<int> getOutputOffset(const vector<int>& outputTensorSize, const vector<int>& center);

    template<typename OtherVoxelType> void copyImagePropertyFrom(ITKImageIO<OtherVoxelType, Dimension> &other);

    typename ImageType::PointType m_origin;
    typename ImageType::SizeType m_imageSize;
    typename ImageType::SpacingType m_spacing;
    typename ImageType::DirectionType m_direction;
};

#include "ITKImageIO.hpp"


#endif //CDLF_FRAMEWORK_ITK_RWRITER_H
