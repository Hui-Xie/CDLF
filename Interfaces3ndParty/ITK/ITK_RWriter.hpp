//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ITK_RWriter.h"

template <typename VoxelType, int Dimension>
ITK_RWriter<VoxelType, Dimension>::ITK_RWriter(){

}

template <typename VoxelType, int Dimension>
ITK_RWriter<VoxelType, Dimension>::~ITK_RWriter(){

}

template <typename VoxelType, int Dimension>
void ITK_RWriter<VoxelType, Dimension>::readFile(const string& filename, Tensor<float>*& pTensor ){
    using ImageType = itk::Image< VoxelType, Dimension >;
    using ReaderType = itk::ImageFileReader< ImageType >;
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( filename );
    reader->Update();
    typename ImageType::Pointer image = reader->GetOutput();


}


template <typename VoxelType, int Dimension>
void ITK_RWriter<VoxelType, Dimension>::writeFile(const Tensor<float>* pTensor, const string& filename){

}