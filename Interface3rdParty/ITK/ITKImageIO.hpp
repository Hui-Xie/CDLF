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
   //null
}

template <typename VoxelType, int Dimension>
ITKImageIO<VoxelType, Dimension>::~ITKImageIO(){

}

template <typename VoxelType, int Dimension>
void ITKImageIO<VoxelType, Dimension>::readFile(const string& filename, Tensor<VoxelType>*& pTensor ){
    using ReaderType = itk::ImageFileReader< ImageType >;
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( filename );
    reader->Update();
    typename ImageType::Pointer image = reader->GetOutput();

    // get ImageSize
    typename ImageType::RegionType region = image->GetLargestPossibleRegion();
    m_imageSize =region.GetSize();
    const int dim = m_imageSize.GetSizeDimension();
    vector<int> tensorSize;
    for (int i=0; i<dim; ++i){
        tensorSize.push_back(m_imageSize[i]);
    }
    tensorSize = reverseVector(tensorSize);
    pTensor = new Tensor<VoxelType>(tensorSize);

    //get Image origin, spacing etc
    m_origin = image->GetOrigin();
    m_spacing = image->GetSpacing();
    m_direction = image->GetDirection();

    itk::ImageRegionConstIteratorWithIndex<ImageType> iter(image,region);
    iter.GoToBegin();
    vector<int> tensorIndex(dim,0);
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

template<typename VoxelType, int Dimension>
void ITKImageIO<VoxelType, Dimension>::readLabelFileAndOrigin(const string &labelFilename, Tensor<VoxelType> *&pTensor,
                                                              typename itk::Image<VoxelType, Dimension>::PointType& labelOrigin) const {

    using ReaderType = itk::ImageFileReader< ImageType >;
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( labelFilename );
    reader->Update();
    typename ImageType::Pointer image = reader->GetOutput();

    // get ImageSize
    typename ImageType::RegionType region  = image->GetLargestPossibleRegion();
    typename ImageType::SizeType imageSize = region.GetSize();
    const int dim = imageSize.GetSizeDimension();
    vector<int> tensorSize;
    for (int i=0; i<dim; ++i){
        tensorSize.push_back(imageSize[i]);
    }
    tensorSize = reverseVector(tensorSize);
    pTensor = new Tensor<VoxelType>(tensorSize);
    pTensor->zeroInitialize();

    //get Image origin, spacing etc
    labelOrigin = image->GetOrigin();

    //do not need the 2 lines below:
    //typename ImageType::SpacingType spacing = image->GetSpacing();
    //typename ImageType::DirectionType direction = image->GetDirection();

    itk::ImageRegionConstIteratorWithIndex<ImageType> iter(image,region);
    iter.GoToBegin();
    vector<int> tensorIndex(dim,0);
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
template<typename OutputValueType>
void ITKImageIO<VoxelType, Dimension>::writeFileWithSameInputDim(const Tensor<OutputValueType>* pTensor, const vector<int>& offset,
                                                  const string& filename)
{
    vector<int> tensorSize = reverseVector(pTensor->getDims());
    const int dim = tensorSize.size();
    if (dim != m_imageSize.GetSizeDimension()){
        cout<<"Error: the output tensor has different dimension with the input image."<<endl;
        return;
    }

    using OutputImageType = itk::Image< OutputValueType, Dimension >;

    typename OutputImageType::RegionType region;
    typename OutputImageType::IndexType start;
    typename OutputImageType::SizeType size;
    typename OutputImageType::PointType newOrigin;
    for(int i=0; i< dim; ++i){
        start[i] = 0;
        size[i] = tensorSize[i];
        newOrigin[i] = m_origin[i]+offset[i]*m_spacing[i]*m_direction[i][i];
    }
    region.SetSize(size);
    region.SetIndex(start);

    typename OutputImageType::Pointer image = OutputImageType::New();
    image->SetRegions(region);
    image->Allocate();

    //set origin, spacing, cosine matrix
    image->SetSpacing( m_spacing );
    image->SetOrigin(newOrigin);
    image->SetDirection(m_direction);

    //copy image data
    int i=0;
    int N = pTensor->getLength();
    typename  itk::ImageRegionConstIteratorWithIndex<OutputImageType>::IndexType itkIndex;
    while(i<N)
    {
        vector<int> tensorIndex = pTensor->offset2Index(i);
        for (int k=0;k<dim;++k){
            itkIndex[dim-1-k]= tensorIndex[k];
        }
        image->SetPixel(itkIndex, pTensor->e(i));
        ++i;
    }


    typedef itk::ImageFileWriter<OutputImageType> WriterType;
    typename WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(image);
    writer->Update();
    cout<<"Info: An output image at "<<filename<<endl;
}

template <typename VoxelType, int Dimension>
template<typename OutputValueType>
void ITKImageIO<VoxelType, Dimension>::writeFileWithLessInputDim(const Tensor<OutputValueType>* pTensor, const vector<int>& offset,
                                                                 const string& filename)
{
    vector<int> tensorSize = reverseVector(pTensor->getDims());
    const int dim = tensorSize.size();
    if (dim +1 != m_imageSize.GetSizeDimension()){
        cout<<"Error: the output tensor should has One less dimension than the input image."<<endl;
        return;
    }
    using LessImageType = itk::Image< OutputValueType, Dimension-1>;

    typename LessImageType::RegionType region;
    typename LessImageType::IndexType start;
    typename LessImageType::SizeType size;
    typename LessImageType::PointType newOrigin;
    for(int i=0; i< dim; ++i){
        start[i] = 0;
        size[i] = tensorSize[i];
        newOrigin[i] = m_origin[i]+offset[i]*m_spacing[i]*m_direction[i][i];
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
        for (int j=0; j<dim; ++j){
            newDirection[i][j] = m_direction[i][j];
        }
    }

    image->SetSpacing( newSpacing );
    image->SetOrigin(newOrigin);
    image->SetDirection(newDirection);

    //copy image data
    int i=0;
    int N = pTensor->getLength();
    typename  itk::ImageRegionConstIteratorWithIndex<LessImageType>::IndexType itkIndex;
    while(i<N)
    {
        vector<int> tensorIndex = pTensor->offset2Index(i);
        for (int k=0;k<dim;++k){
            itkIndex[dim-1-k]= tensorIndex[k];
        }
        image->SetPixel(itkIndex, pTensor->e(i));
        ++i;
    }

    typedef itk::ImageFileWriter<LessImageType> WriterType;
    typename WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(image);
    writer->Update();
    cout<<"Info: An output image at "<<filename<<endl;
}

template<typename VoxelType, int Dimension>
void ITKImageIO<VoxelType, Dimension>::extendLabelFileVolume(const string &labelFilename, Tensor<VoxelType> *&pTensor) {
    Tensor<VoxelType>* pSmallTensor = nullptr;
    typename itk::Image<VoxelType, Dimension>::PointType labelOrigin;
    readLabelFileAndOrigin(labelFilename, pSmallTensor,labelOrigin);

    //compute offset
    vector<int> offsetVec(Dimension, 0);
    for(int i=0; i<Dimension; ++i){
        offsetVec[i] = (int)((labelOrigin[i]- m_origin[i])/(m_spacing[i]*m_direction[i][i]) +0.5);
    }
    offsetVec = reverseVector(offsetVec);

    vector<int> tensorSize;
    for (int i=0; i<Dimension; ++i){
        tensorSize.push_back(m_imageSize[Dimension-1-i]);  //reverse dims
    }
    pTensor = new Tensor<VoxelType> (tensorSize);
    pTensor->zeroInitialize();
    pSmallTensor->putInBiggerTensor(pTensor, offsetVec, 1);

    if (nullptr != pSmallTensor){
        delete pSmallTensor;
    }
}

template<typename VoxelType, int Dimension>
vector<int> ITKImageIO<VoxelType, Dimension>::getOutputOffset(const vector<int> outputTensorSize) {
    vector<int> offset(Dimension, 0);
    for(int i=0; i< Dimension; ++i){
        offset[i] = (m_imageSize[Dimension-1-i] - outputTensorSize[i])/2;
    }
    return offset;
}

template<typename VoxelType, int Dimension>
template<typename OtherVoxelType>
void ITKImageIO<VoxelType, Dimension>::copyImagePropertyFrom(ITKImageIO<OtherVoxelType, Dimension> &other) {
    for (int i=0; i<Dimension; ++i){
       m_origin[i] = other.m_origin[i];
       m_imageSize[i] = other.m_imageSize[i];
       m_spacing[i] = other.m_spacing[i];
       for (int j= 0; j<Dimension; ++j){
           m_direction[i][j] = other.m_direction[i][j];
       }
    }
}


