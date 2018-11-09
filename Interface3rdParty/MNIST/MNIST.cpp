//
// Created by Hui Xie on 8/8/2018.
//

#include "MNIST.h"
#include <fstream>


MNIST::MNIST(const string &mnistDir, bool onlyTestSet) {
    m_mnistDir = mnistDir;
    m_trainImageFile = m_mnistDir + "/train-images-idx3-ubyte";
    m_trainLabelFile = m_mnistDir + "/train-labels-idx1-ubyte";
    m_testImageFile = m_mnistDir + "/t10k-images-idx3-ubyte";
    m_testLabelFile = m_mnistDir + "/t10k-labels-idx1-ubyte";

    m_pTrainImages = nullptr;
    m_pTrainLabels = nullptr;
    m_pTestImages = nullptr;
    m_pTestLabels = nullptr;

    m_partDigits = {0,1}; //user can change number of elements to many, at least 2.

    m_pTrainImagesPart = nullptr;
    m_pTrainLabelsPart = nullptr;
    m_pTestImagesPart = nullptr;
    m_pTestLabelsPart = nullptr;

    m_bOnlyTestSet = onlyTestSet;
}

MNIST::~MNIST() {
    deleteWholeDataSet();
    deletePartDataSet();
}

void MNIST::deleteWholeDataSet(){
    if (nullptr != m_pTrainImages) {
        delete m_pTrainImages;
        m_pTrainImages = nullptr;
    }

    if (nullptr != m_pTrainLabels) {
        delete m_pTrainLabels;
        m_pTrainLabels = nullptr;
    }

    if (nullptr != m_pTestImages) {
        delete m_pTestImages;
        m_pTestImages = nullptr;
    }

    if (nullptr != m_pTestLabels) {
        delete m_pTestLabels;
        m_pTestLabels = nullptr;
    }
}

void MNIST::deletePartDataSet(){
    if (nullptr != m_pTrainImagesPart) {
        delete m_pTrainImagesPart;
        m_pTrainImagesPart = nullptr;
    }

    if (nullptr != m_pTrainLabelsPart) {
        delete m_pTrainLabelsPart;
        m_pTrainLabelsPart = nullptr;
    }

    if (nullptr != m_pTestImagesPart) {
        delete m_pTestImagesPart;
        m_pTestImagesPart = nullptr;
    }

    if (nullptr != m_pTestLabelsPart) {
        delete m_pTestLabelsPart;
        m_pTestLabelsPart = nullptr;
    }
}

long MNIST::hexChar4ToLong(char *buff) {
    long temp = 0;
    for (int i = 0; i < 4; ++i) {
        temp += ((unsigned char) buff[i]) * pow(16, (3 - i) * 2);
    }
    return temp;
}

bool MNIST::isDigitInVector(const unsigned char digit){
    const int N = m_partDigits.size();
    for (int i=0; i<N; ++i){
        if (digit == m_partDigits[i]) return true;
    }
    return false;
}

void MNIST::extractPart(const Tensor<unsigned char> * pWholeImages,  const Tensor<unsigned char> * pWholeLabels,
                 Tensor<unsigned char> * pPartImages,  Tensor<unsigned char> *  pPartLabels){
    long N = pWholeLabels->getLength();
    long NPart = 0;
    long imageSize = 28*28*sizeof(unsigned char);
    for(long i=0; i<N; ++i){
        if (isDigitInVector(pWholeLabels->e(i))) {
            pPartLabels->e(NPart) = pWholeLabels->e(i);
            memcpy(pPartImages->getData()+NPart*imageSize, pWholeImages->getData()+i*imageSize, imageSize );
            ++NPart;
        }
    }
}

int MNIST::readIdxFile(const string &fileName, Tensor<unsigned char> *&pTensor) {
    ifstream ifs(fileName, ifstream::in | ifstream::binary);
    ifs.seekg(ios_base::beg);
    if (!ifs.good()) {
        ifs.close();
        cout << "Error: read file error: " << fileName << endl;
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

        } else if (0x01 == magicNum[3]) {// Label file
            ifs.read(dim, 4);
            numImages = hexChar4ToLong(dim);
            isImage = false;
        } else {
            cout << "Error: incorrect magic number in Idx file. Exit." << endl;
            ifs.close();
            return 2;
        }
    } else {
        cout << "Error: incorrect Idx file. Exit." << endl;
        ifs.close();
        return 3;
    }

    if (isImage) {
        pTensor = new Tensor<unsigned char>({numImages, rows, cols});
    } else {
        pTensor = new Tensor<unsigned char>({numImages, 1});
    }
    long numBytes = pTensor->getLength() * sizeof(unsigned char);
    char *buff = new char[numBytes];
    ifs.read(buff, numBytes);
    pTensor->copyDataFrom(buff, numBytes);
    delete[] buff;

    ifs.close();
    return 0;
}

void MNIST::loadData() {
    if (!m_bOnlyTestSet) {
        readIdxFile(m_trainImageFile, m_pTrainImages);
        readIdxFile(m_trainLabelFile, m_pTrainLabels);
        cout << "Info: read " << m_pTrainImages->getDims()[0] << " training images. " << endl;
    }

    readIdxFile(m_testImageFile, m_pTestImages);
    readIdxFile(m_testLabelFile, m_pTestLabels);
    cout<<"Info: read "<<m_pTestImages->getDims()[0] <<" test images. "<<endl;
}

void MNIST::tailorData(){
    //get the total number of part train dataset
    long N = m_pTrainLabels->getLength();
    long NTrainPart = 0;
    for (long i=0;i<N;++i){
        if (isDigitInVector(m_pTrainLabels->e(i))){
            ++NTrainPart;
        }
    }
    cout<<"Info: MNIST tain part dataset has total "<<NTrainPart<<" elements."<<endl;
    m_pTrainImagesPart = new  Tensor<unsigned char> ({NTrainPart,28,28});
    m_pTrainLabelsPart = new  Tensor<unsigned char> ({NTrainPart,1});
    extractPart(m_pTrainImages,m_pTrainLabels,m_pTrainImagesPart, m_pTrainLabelsPart);

    //get the total number of part test dataset
    N = m_pTestLabels->getLength();
    long NTestPart = 0;
    for (long i=0;i<N;++i){
        if (isDigitInVector(m_pTestLabels->e(i))){
            ++NTestPart;
        }
    }
    cout<<"Info: MNIST test part dataset has total "<<NTestPart<<" elements."<<endl;
    m_pTestImagesPart = new  Tensor<unsigned char> ({NTestPart,28,28});
    m_pTestLabelsPart = new  Tensor<unsigned char> ({NTestPart,1});
    extractPart(m_pTestImages,m_pTestLabels,m_pTestImagesPart, m_pTestLabelsPart);
}

void MNIST::displayImage(Tensor<unsigned char> *pImages, const long index) {
    Tensor<unsigned char> slice = pImages->slice(index);
    slice.printElements(true);
}




