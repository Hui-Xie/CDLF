//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ITK_RWriter.h"
#include <vector>
#include "itkImageRegionConstIterator.h"

template <typename VoxelType, int Dimension>
ITK_RWriter<VoxelType, Dimension>::ITK_RWriter(){

}

template <typename VoxelType, int Dimension>
ITK_RWriter<VoxelType, Dimension>::~ITK_RWriter(){

}

template <typename VoxelType, int Dimension>
void ITK_RWriter<VoxelType, Dimension>::readFile(const string& filename, Tensor<float>*& pTensor ){
    using ReaderType = itk::ImageFileReader< ImageType >;
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( filename );
    reader->Update();
    typename ImageType::Pointer image = reader->GetOutput();

    // get ImageSize
    typename ImageType::RegionType region = image->GetLargestPossibleRegion();
    m_imageSize =region.GetSize();
    const int dim = m_imageSize.GetSizeDimension();
    vector<long> tensorSize;
    for (int i=0; i<dim; ++i){
        tensorSize.push_back(m_imageSize[i]);
    }
    pTensor = new Tensor<float>(tensorSize);

    //get Image origin, spacing etc
    m_origin = image->GetOrigin();
    m_spacing = image->GetSpacing();

    itk::ImageRegionConstIterator<ImageType> iter(image,region);
    long i=0;
    while(!iter.IsAtEnd())
    {
        pTensor->e(i)= iter.Get();
        ++iter;
        ++i;
    }
}


template <typename VoxelType, int Dimension>
void ITK_RWriter<VoxelType, Dimension>::writeFile(const Tensor<float>* pTensor, const string& filename){

}