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
        cout<<"Error: read "<<filename<<" in tipl::io::nifti::load_from_file()"<<endl;
        return;
    };
    parser >> imageData;

    m_imageHeader1 = parser.nif_header;
    m_imageHeader2 = parser.nif_header2;
    const unsigned int dim = imageData.dimension;


    vector<long> tensorSize(dim,0);
    for(int i=0; i<dim; ++i){
        tensorSize[i] = m_imageHeader2.dim[dim-i];
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
                pTensor->e(i,j,k) = (float)imageData.at(k,j,i);

    }
    else{
        cout<<"Error: the input NIfTI file has incorrect dimension"<<endl;

    }
}

template<typename VoxelType, int Dimension>
void TIPLIO<VoxelType, Dimension>::write3DNIfTIFile(const Tensor<float> *pTensor, const vector<long> &offset,
                                                    const string &filename) {
    const vector<long> tensorSize = pTensor->getDims();
    tipl::io::nifti parser;
    const unsigned int dim = tensorSize.size();

    if (dim + 1 != m_imageHeader2.dim[0]) {
        cout << "Error: output image has different dimension with input image" << endl;
        return;
    }

    for (int i = 0; i < dim; ++i) {
        m_imageHeader2.dim[dim - i] = tensorSize[i];
    }

    // modify origin,
    if (m_imageHeader1.qform_code > 0){
        m_imageHeader1.qoffset_x += offset[2]*m_imageHeader1.pixdim[1];
        m_imageHeader1.qoffset_y += offset[1]*m_imageHeader1.pixdim[2];
        m_imageHeader1.qoffset_z += offset[0]*m_imageHeader1.pixdim[3];
    }
    if (m_imageHeader1.sform_code > 0){
        m_imageHeader1.srow_x[3] += offset[2]*m_imageHeader1.pixdim[1];
        m_imageHeader1.srow_y[3] += offset[1]*m_imageHeader1.pixdim[2];
        m_imageHeader1.srow_z[3] += offset[0]*m_imageHeader1.pixdim[3];
    }
    //copy headers
    parser.nif_header = m_imageHeader1;
    parser.nif_header2 = m_imageHeader2;


    tipl::image<VoxelType, Dimension> imageData(tipl::geometry<Dimension>(tensorSize[2], tensorSize[1], tensorSize[0]));
    for (long i = 0; i < tensorSize[0]; ++i)
        for (long j = 0; j < tensorSize[1]; ++j)
            for (long k = 0; k < tensorSize[2]; ++k) {
                imageData.at(k, j, i) = (VoxelType) pTensor->e(i, j, k);
            }
    parser.toLPS(imageData);
    parser << imageData;

    //save file
    parser.save_to_file(filename.c_str());
    cout << "Infor:  " << filename << "  ouptput." << endl;

}