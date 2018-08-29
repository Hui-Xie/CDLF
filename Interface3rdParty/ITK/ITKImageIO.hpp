//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ITKImageIO.h"
#include <vector>
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

template <typename VoxelType, int Dimension>
ITKImageIO<VoxelType, Dimension>::ITKImageIO(){
    m_dim = Dimension;
}

template <typename VoxelType, int Dimension>
ITKImageIO<VoxelType, Dimension>::~ITKImageIO(){

}

template <typename VoxelType, int Dimension>
void ITKImageIO<VoxelType, Dimension>::readFile(const string& filename, Tensor<float>*& pTensor ){
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
    tensorSize = reverseVector(tensorSize);
    pTensor = new Tensor<float>(tensorSize);

    //get Image origin, spacing etc
    m_origin = image->GetOrigin();
    m_spacing = image->GetSpacing();
    m_direction = image->GetDirection();

    itk::ImageRegionConstIteratorWithIndex<ImageType> iter(image,region);
    iter.GoToBegin();
    vector<long> tensorIndex(dim,0);
    while(!iter.IsAtEnd())
    {
        typename  itk::ImageRegionConstIteratorWithIndex<ImageType>::IndexType index = iter.GetIndex();
        for (int k=0;k<dim;++k){
            tensorIndex[k] = index[dim-1-k];
        }
        pTensor->e(tensorIndex)= (float)iter.Get();
        ++iter;
    }

}


template <typename VoxelType, int Dimension>
void ITKImageIO<VoxelType, Dimension>::writeFileWithSameInputDim(const Tensor<float>* pTensor, const vector<long>& offset,
                                                  const string& filename)
{
    vector<long> tensorSize = reverseVector(pTensor->getDims());
    const int dim = tensorSize.size();
    if (dim != m_imageSize.GetSizeDimension()){
        cout<<"Error: the output tensor has different dimension with the input image."<<endl;
        return;
    }

    typename ImageType::RegionType region;
    typename ImageType::IndexType start;
    typename ImageType::SizeType size;
    typename ImageType::PointType newOrigin;
    for(int i=0; i< dim; ++i){
        start[i] = 0;
        size[i] = tensorSize[i];
        newOrigin[i] = m_origin[i]+offset[i]*m_spacing[i];
    }
    region.SetSize(size);
    region.SetIndex(start);

    typename ImageType::Pointer image = ImageType::New();
    image->SetRegions(region);
    image->Allocate();

    //set origin, spacing, cosine matrix
    image->SetSpacing( m_spacing );
    image->SetOrigin(newOrigin);
    image->SetDirection(m_direction);

    //copy image data
    long i=0;
    long N = pTensor->getLength();
    typename  itk::ImageRegionConstIteratorWithIndex<ImageType>::IndexType itkIndex;
    while(i<N)
    {
        vector<long> tensorIndex = pTensor->offset2Index(i);
        for (int k=0;k<dim;++k){
            itkIndex[dim-1-k]= tensorIndex[k];
        }
        image->SetPixel(itkIndex, (VoxelType)pTensor->e(i));
        ++i;
    }

    typedef itk::ImageFileWriter<ImageType> WriterType;
    typename WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(image);
    writer->Update();
    cout<<"Info: An output image "<<filename<<" output"<<endl;
}

template <typename VoxelType, int Dimension>
void ITKImageIO<VoxelType, Dimension>::writeFileWithLessInputDim(const Tensor<float>* pTensor, const vector<long>& offset,
                                                                 const string& filename)
{
    vector<long> tensorSize = reverseVector(pTensor->getDims());
    const int dim = tensorSize.size();
    if (dim +1 != m_imageSize.GetSizeDimension()){
        cout<<"Error: the output tensor should has One less dimension than the input image."<<endl;
        return;
    }

    using LessImageType = itk::Image< VoxelType, Dimension-1>;

    typename LessImageType::RegionType region;
    typename LessImageType::IndexType start;
    typename LessImageType::SizeType size;
    typename LessImageType::PointType newOrigin;
    for(int i=0; i< dim; ++i){
        start[i] = 0;
        size[i] = tensorSize[i];
        newOrigin[i] = m_origin[i]+offset[i]*m_spacing[i];
    }
    region.SetSize(size);
    region.SetIndex(start);

    typename LessImageType::Pointer image = LessImageType::New();
    image->SetRegions(region);
    image->Allocate();

    //set origin, spacing, cosine matrix
    typename LessImageType::SpacingType newSpacing;
    typename LessImageType::DirectionType newDirection;
    for (int i=0;i<dim; ++i){
        newSpacing[i] = m_spacing[i];
        newDirection[i] = m_direction[i];
     }

    image->SetSpacing( newSpacing );
    image->SetOrigin(newOrigin);
    image->SetDirection(newDirection);

    //copy image data
    long i=0;
    long N = pTensor->getLength();
    typename  itk::ImageRegionConstIteratorWithIndex<LessImageType>::IndexType itkIndex;
    while(i<N)
    {
        vector<long> tensorIndex = pTensor->offset2Index(i);
        for (int k=0;k<dim;++k){
            itkIndex[dim-1-k]= tensorIndex[k];
        }
        image->SetPixel(itkIndex, (VoxelType)pTensor->e(i));
        ++i;
    }

    typedef itk::ImageFileWriter<LessImageType> WriterType;
    typename WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(image);
    writer->Update();
    cout<<"Info: An output image "<<filename<<" output"<<endl;
}