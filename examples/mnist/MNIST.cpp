//
// Created by Sheen156 on 8/8/2018.
//

#include "MNIST.h"
#include <fstream>


MNIST::MNIST(const string& MnistDir){
    m_mnistDir = MnistDir;
    m_trainImageFile = m_mnistDir + "\\train-images.idx3-ubyte";
    m_trainLabelFile = m_mnistDir + "\\train-labels.idx1-ubyte";
    m_testImageFile =  m_mnistDir +"\\t10k-images.idx3-ubyte";
    m_testLabelFile = m_mnistDir + "\\t10k-labels.idx1-ubyte";

    m_pTrainImages = nullptr;
    m_pTrainLabels = nullptr;
    m_pTestImages = nullptr;
    m_pTestLabels = nullptr;
}
MNIST::~MNIST(){
    if (nullptr != m_pTrainImages){
        delete m_pTrainImages;
        m_pTrainImages = nullptr;
    }

    if (nullptr != m_pTrainLabels){
        delete m_pTrainLabels;
        m_pTrainLabels = nullptr;
    }

    if (nullptr != m_pTestImages){
        delete m_pTestImages;
        m_pTestImages = nullptr;
    }

    if (nullptr != m_pTestLabels){
        delete m_pTestLabels;
        m_pTestLabels = nullptr;
    }
}

long MNIST::hexChar4ToLong(char *buff){
    long  temp =0;
    for (int i=0; i<4; ++i){
        temp += ((unsigned char)buff[i])* pow(16,(3-i)*2);
    }
    return temp;
}

int MNIST::readIdxFile(const string &fileName, Tensor<unsigned char>* &pTensor){
    ifstream ifs(fileName, ifstream::in | ifstream::binary);
    ifs.seekg(ios_base::beg);
    if (!ifs.good()) {
        ifs.close();
        cout << "Error: read file error: " << fileName<<endl;
        return 1;
    }

    long numImages = 0;
    long rows = 0;
    long cols = 0;
    bool isImage = true; //False is a Label file

    //read magic number and dimension
    char magicNum[4];
    char dim[4];
    ifs.read(magicNum, 4);
    if (0x00 == magicNum[0] && 0x00 == magicNum[1] && 0x08 == magicNum[2]) {
        if (0x03 == magicNum[3]) {//Image file
            ifs.read(dim, 4);
            numImages = hexChar4ToLong(dim);
            ifs.read(dim, 4);
            rows = hexChar4ToLong(dim);
            ifs.read(dim, 4);
            cols = hexChar4ToLong(dim);
            isImage = true;

        }else if (0x01 == magicNum[3]) {// Label file
            ifs.read(dim, 4);
            numImages = hexChar4ToLong(dim);
            isImage = false;
        }
        else{
            cout << "Error: incorrect magic number in Idx file. Exit." << endl;
            ifs.close();
            return 2;
        }
    } else {
        cout << "Error: incorrect Idx file. Exit." << endl;
        ifs.close();
        return 3;
    }

    if (isImage){
        pTensor = new Tensor<unsigned char>({numImages,rows, cols});
    }
    else{
        pTensor = new Tensor<unsigned char>({numImages, 1});
    }
    long numBytes = pTensor->getLength()* sizeof(unsigned char);
    char* buff = new char[numBytes];
    ifs.read(buff, numBytes);
    pTensor->copyDataFrom(buff, numBytes);
    delete[] buff;

    ifs.close();
    return 0;
}

void MNIST::loadData(){
    readIdxFile(m_trainImageFile,m_pTrainImages);
    readIdxFile(m_trainLabelFile,m_pTrainLabels);
    readIdxFile(m_testImageFile,m_pTestImages);
    readIdxFile(m_testLabelFile,m_pTestLabels);
}

void MNIST::displayImage(Tensor<unsigned char>* pImages, const long index){
    Tensor<unsigned char> slice = pImages->slice(index);
    slice.printElements(true);
}