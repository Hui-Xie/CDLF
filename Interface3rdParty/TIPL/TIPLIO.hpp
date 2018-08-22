//
// Created by hxie1 on 8/22/18.
//

#include "TIPLIO.h"



template <typename VoxelType, int Dimension>
TIPLIO<VoxelType, Dimension>::TIPLIO(){

}

template <typename VoxelType, int Dimension>
TIPLIO<VoxelType, Dimension>::~TIPLIO(){

}

template <typename VoxelType, int Dimension>
void TIPLIO<VoxelType, Dimension>::readNIfTIFile(const string & filename, Tensor<float>*& pTensor){
    tipl::io::nifti parser;
    tipl::image<VoxelType,Dimension> imageData;
    if (!parser.load_from_file(filename)){
        cout<<"Error: read "<<filename<<endl;
        return;
    };
    parser >> imageData;

    m_imageHeader = parser.nif_header;
    const unsigned int dim = imageData.dimension;


    vector<long> tensorSize(dim,0);
    for(int i=0; i<dim; ++i){
        tensorSize[i] = parser.nif_header2.dim[dim-i];
    }
    pTensor = new Tensor<float>(tensorSize);

    if (2 == dim){
        for (long i=0; i<tensorSize[0]; ++i)
            for (long j=0; j<tensorSize[1];++j)
                pTensor->e(i,j) = imageData.at(j,i);

    }
    else if (3 == dim){
        for (long i=0; i<tensorSize[0]; ++i)
            for (long j=0; j<tensorSize[1];++j)
                for (long k=0; k<tensorSize[2];++k)
                pTensor->e(i,j,k) = imageData.at(k,j,i);

    }
    else{
        cout<<"Error: the input NIfTI file has incorrect dimension"<<endl;

    }
}

template <typename VoxelType, int Dimension>
void TIPLIO<VoxelType, Dimension>::writeNIfTIFile(const Tensor<float>* pTensor, const vector<long>& offset, const string & filename){

}