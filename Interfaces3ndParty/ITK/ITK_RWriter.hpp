//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ITK_RWriter.h"
#include <vector>
#include "itkImageRegionConstIterator.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

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
    iter.GoToBegin();
    while(!iter.IsAtEnd())
    {
        pTensor->e(i)= iter.Get();
        ++iter;
        ++i;
    }

}


template <typename VoxelType, int Dimension>
void ITK_RWriter<VoxelType, Dimension>::writeFile(const Tensor<float> *pTensor, const vector<long> &sizeOffset,
                                                  const string &filename) {
    vector<long> tensorSize = pTensor->getDims();
    int dim = tensorSize.size();

    typename ImageType::RegionType region;
    typename ImageType::IndexType start;
    typename ImageType::SizeType size;
    for(int i=0; i< dim; ++i){
        start[i] = 0;
        size[i] = tensorSize[i];
    }
    region.SetSize(size);
    region.SetIndex(start);

    typename ImageType::Pointer image = ImageType::New();
    image->SetRegions(region);
    image->Allocate();

    //Todo: set origin and spacing


    //Todo: copy data

    typedef itk::ImageFileWriter<ImageType> WriterType;
    typename WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(image);
    writer->Update();
}