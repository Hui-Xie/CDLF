//
// Created by hxie1 on 8/22/18.
//

#include "TIPLIO.h"
#include "tipl/tipl.hpp"


template <typename VoxelType, int Dimension>
TIPLIO<VoxelType, Dimension>::TIPLIO(){

}

template <typename VoxelType, int Dimension>
TIPLIO<VoxelType, Dimension>::~TIPLIO(){

}

template <typename VoxelType, int Dimension>
void TIPLIO<VoxelType, Dimension>::readNIfTIFile(const string & filename, Tensor<float>*& pTensor){
    tipl::io::nifti parser;
    tipl::image<VoxelType,Dimension> image_data;
    if (!parser.load_from_file(filename)){
        cout<<"Error: read "<<filename<<endl;
        return;
    };
    parser >> image_data;

    cout<<parser.nif_header.descrip<<endl;



}

template <typename VoxelType, int Dimension>
void TIPLIO<VoxelType, Dimension>::writeNIfTIFile(const Tensor<float>* pTensor, const vector<long>& offset, const string & filename){

}