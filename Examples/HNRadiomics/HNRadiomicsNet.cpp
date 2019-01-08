
#include "HNRadiomicsNet.h"


HNRadiomicsNet::HNRadiomicsNet(const string &name, const string &saveDir) : FeedForwardNet(name, saveDir) {
    m_pDataMgr = nullptr;
}

HNRadiomicsNet::~HNRadiomicsNet() {
  //null
}

void HNRadiomicsNet::build() {
   //null: use csv file to create network
}

void HNRadiomicsNet::train() {
    InputLayer *inputLayer = getInputLayer();
    MeanSquareLossLayer *lossLayer = (MeanSquareLossLayer *) getFinalLayer();

    const int N =m_pDataMgr->m_NTrainFile;
    const int batchSize = getBatchSize();
    const float learningRate = getLearningRate();
    const int numBatch = (N + batchSize -1) / batchSize;
    int nIter = 0;
    int nBatch = 0;
    vector<int> randSeq = generateRandomSequence(N);
    while (nBatch < numBatch) {
        zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < N; ++i) {

            const string imageFilePath = m_pDataMgr->m_trainImagesVector[randSeq[nIter]];
            const string labelFilePath = m_pDataMgr->getLabelPathFrom(imageFilePath);

            Tensor<float>* pImage = nullptr;

            m_pDataMgr->readTrainImageFile(randSeq[nIter], pImage);
            Tensor<float>* pSubImage = new Tensor<float>(inputLayer->m_tensorSize);
            pImage->subTensorFromTopLeft((pImage->getDims() - pSubImage->getDims())/2, pSubImage, 1);
            inputLayer->setInputTensor(*pSubImage);
            if (nullptr != pImage) {
                delete pImage;
                pImage = nullptr;
            }
            if (nullptr != pSubImage) {
                delete pSubImage;
                pSubImage = nullptr;
            }


            m_pDataMgr->readLabelFile(labelFilePath, pImage);
            Tensor<float>* pSubLabel = new Tensor<float>(lossLayer->m_prevLayer->m_tensorSize);
            pImage->subTensorFromTopLeft((pImage->getDims() - pSubLabel->getDims())/2, pSubLabel, 1);
            lossLayer->setGroundTruth(*pSubLabel);
            if (nullptr != pImage) {
                delete pImage;
                pImage = nullptr;
            }
            if (nullptr != pSubLabel) {
                delete pSubLabel;
                pSubLabel = nullptr;
            }

            forwardPropagate();
            backwardPropagate(true);
            ++nIter;
        }
        sgd(learningRate, i);
        ++nBatch;
    }
}

float HNRadiomicsNet::test() {
    InputLayer *inputLayer = getInputLayer();
    MeanSquareLossLayer *lossLayer = (MeanSquareLossLayer *) getFinalLayer();

    int n = 0;
    const int N = m_pDataMgr->m_NTestFile;
    float loss = 0.0;
    while (n < N) {

        const string imageFilePath = m_pDataMgr->m_testImagesVector[n];
        const string labelFilePath = m_pDataMgr->getLabelPathFrom(imageFilePath);

        Tensor<float>* pImage = nullptr;

        m_pDataMgr->readTestImageFile(n, pImage);
        Tensor<float>* pSubImage = new Tensor<float>(inputLayer->m_tensorSize);
        pImage->subTensorFromTopLeft((pImage->getDims() - pSubImage->getDims())/2, pSubImage, 1);
        inputLayer->setInputTensor(*pSubImage);
        if (nullptr != pImage) {
            delete pImage;
            pImage = nullptr;
        }
        if (nullptr != pSubImage) {
            delete pSubImage;
            pSubImage = nullptr;
        }


        m_pDataMgr->readLabelFile(labelFilePath, pImage);
        Tensor<float>* pSubLabel = new Tensor<float>(lossLayer->m_prevLayer->m_tensorSize);
        pImage->subTensorFromTopLeft((pImage->getDims() - pSubLabel->getDims())/2, pSubLabel, 1);
        lossLayer->setGroundTruth(*pSubLabel);
        if (nullptr != pImage) {
            delete pImage;
            pImage = nullptr;
        }
        if (nullptr != pSubLabel) {
            delete pSubLabel;
            pSubLabel = nullptr;
        }

        forwardPropagate();
        loss += lossLayer->getLoss();
        ++n;

    }
    return  loss/N;

}
