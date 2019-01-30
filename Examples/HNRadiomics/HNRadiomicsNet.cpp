
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

void HNRadiomicsNet::defineAssemblyLoss() {
    AssemblyLossLayer *lossLayer = (AssemblyLossLayer *) getFinalLayer();
    Layer* prevLayer = lossLayer->m_prevLayer;
    lossLayer->addLoss( new SquareLossLayer(-1, "SquareLoss", prevLayer, 1));
    lossLayer->addLoss( new DiceLossLayer(-2, "DiceLoss", prevLayer));
    lossLayer->addLoss( new CrossEntropyLoss(-3, "CrossEntropyLoss", prevLayer));
}

void HNRadiomicsNet::train() {
    InputLayer *inputLayer = getInputLayer();
    AssemblyLossLayer *lossLayer = (AssemblyLossLayer *) getFinalLayer();

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
            if (0 == pSubLabel->L2Norm()){
                cout<<"Error: label file has full zero: "<<labelFilePath<<endl;
                return;
            }
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

            //saveYTensor();
            //savedYTensor();

            ++nIter;
        }
        sgd(learningRate, i);
        ++nBatch;

    }
}

float HNRadiomicsNet::test() {
    InputLayer *inputLayer = getInputLayer();
    AssemblyLossLayer *lossLayer = (AssemblyLossLayer *) getFinalLayer();

    int n = 0;
    const int N = m_pDataMgr->m_NTestFile;
    float loss = 0.0;
    m_dice = 0;
    m_TPR = 0;
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
        if (0 == pSubLabel->L2Norm()){
            cout<<"Error: label file has full zero: "<<labelFilePath<<endl;
            return 0;
        }

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
        m_dice += lossLayer->diceCoefficient(0.5);
        m_TPR  += lossLayer->getTPR(0.5);
        ++n;

    }
    m_dice /= N;
    m_TPR /= N;
    return  loss/N;

}

float HNRadiomicsNet::test(const string &imageFilePath, const string &labelFilePath) {
    InputLayer *inputLayer = getInputLayer();
    AssemblyLossLayer *lossLayer = (AssemblyLossLayer *) getFinalLayer();

    float loss = 0.0;
    m_dice = 0.0;
    m_TPR = 0.0;

    Tensor<float> *pImage = nullptr;
    m_pDataMgr->readImageFile(imageFilePath, pImage);
    Tensor<float> *pSubImage = new Tensor<float>(inputLayer->m_tensorSize);
    pImage->subTensorFromTopLeft((pImage->getDims() - pSubImage->getDims()) / 2, pSubImage, 1);
    inputLayer->setInputTensor(*pSubImage);
    if (nullptr != pImage) {
        delete pImage;
        pImage = nullptr;
    }
    if (nullptr != pSubImage) {
        delete pSubImage;
        pSubImage = nullptr;
    }

    if (!labelFilePath.empty()){
        m_pDataMgr->readLabelFile(labelFilePath, pImage);
        Tensor<float> *pSubLabel = new Tensor<float>(lossLayer->m_prevLayer->m_tensorSize);
        pImage->subTensorFromTopLeft((pImage->getDims() - pSubLabel->getDims()) / 2, pSubLabel, 1);
        lossLayer->setGroundTruth(*pSubLabel);
        if (nullptr != pImage) {
            delete pImage;
            pImage = nullptr;
        }
        if (nullptr != pSubLabel) {
            delete pSubLabel;
            pSubLabel = nullptr;
        }
    }

    forwardPropagate();

    vector<int> offset = m_pDataMgr->getOutputOffset(lossLayer->m_prevLayer->m_tensorSize);

    // output float image for debug
    const string floatImageOutput = m_pDataMgr->generateFloatImagePath(imageFilePath);
    m_pDataMgr->saveImage2File(lossLayer->m_prevLayer->m_pYTensor, offset, floatImageOutput);

    //Output network predicted label
    Tensor<unsigned char> predictResult(lossLayer->m_prevLayer->m_tensorSize);
    lossLayer->getPredictTensor(predictResult, 0.5);
    string outputLabelFilePath = m_pDataMgr->generateLabelFilePath(imageFilePath);
    m_pDataMgr->saveLabel2File(&predictResult, offset, outputLabelFilePath);


    if (!labelFilePath.empty()){
        loss = lossLayer->getLoss();
        m_dice = lossLayer->diceCoefficient(0.5);
        m_TPR = lossLayer->getTPR(0.5);
    }
    return loss;
}
