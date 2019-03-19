
#include "HNSegVNet.h"


HNSegVNet::HNSegVNet(const string &netDir) : FeedForwardNet(netDir) {
    m_pDataMgr = nullptr;


}

HNSegVNet::~HNSegVNet() {
  //null
}

void HNSegVNet::build() {
   //null: use csv file to create network
}

/*
void HNSegVNet::defineAssemblyLoss() {
    DiceLossLayer *lossLayer = (DiceLossLayer *) getFinalLayer();
    Layer* prevLayer = lossLayer->m_prevLayer;
    lossLayer->addLoss( new SquareLossLayer(-1, "SquareLoss", prevLayer, 1));
    lossLayer->addLoss( new CrossEntropyLoss(-2, "CrossEntropyLoss", prevLayer));
    lossLayer->addLoss( new DiceLossLayer(-3, "DiceLoss", prevLayer));
}
*/

void HNSegVNet::setGroundtruth(const string &filename, const vector<float>& radianVec, vector<int>& center, const int translationMaxValue) {
    DiceLossLayer *lossLayer = (DiceLossLayer *) getFinalLayer();

    Tensor<float> *pLabel = nullptr;
    m_pDataMgr->readLabelFile(filename, pLabel);

    Tensor<float> *pRotatedLabel = nullptr;
    pLabel->rotate3D(radianVec, IPPI_INTER_NN, pRotatedLabel);
    center = pRotatedLabel->getCenterOfNonZeroElements();
    randomTranslate(center, translationMaxValue); // support random translation in all axes direction within 15 pixels

    if (nullptr != pLabel) {
        delete pLabel;
        pLabel = nullptr;
    }


    Tensor<float> *pSubLabel = nullptr;
    pSubLabel = new Tensor<float>(lossLayer->m_prevLayer->m_tensorSize);

    //  for lossLayer->m_prevLayer is Softmax
    if (pRotatedLabel->getDims().size() + 1 == lossLayer->m_prevLayer->m_tensorSize.size()) {
        const int k = lossLayer->m_prevLayer->m_tensorSize[0];
        Tensor<float> *pOneHotLabel = nullptr;
        m_pDataMgr->oneHotEncodeLabel(pRotatedLabel, pOneHotLabel, k);
        const vector<int> strideOneHot = vector<int>(lossLayer->m_prevLayer->m_tensorSize.size(), 1);

        //update topLeft index
        vector<int> subImageDims = lossLayer->m_prevLayer->m_tensorSize;
        subImageDims.erase(subImageDims.begin());
        vector<int> topLeft = getTopLeftIndexFrom(pRotatedLabel->getDims(), subImageDims, center);
        topLeft.insert(topLeft.begin(), 0);

        pOneHotLabel->subTensorFromTopLeft(topLeft, pSubLabel, strideOneHot);
        if (nullptr != pOneHotLabel) {
            delete pOneHotLabel;
            pOneHotLabel = nullptr;
        }
    }
        // for lossLayer->m_prevLayer is Sigmoid
    else if (pRotatedLabel->getDims().size() == lossLayer->m_prevLayer->m_tensorSize.size()) {
        const vector<int> stride1 = vector<int>(pRotatedLabel->getDims().size(), 1);
        const vector<int> topLeft = getTopLeftIndexFrom(pRotatedLabel->getDims(),
                                                                    lossLayer->m_prevLayer->m_tensorSize, center);
        pRotatedLabel->subTensorFromTopLeft(topLeft, pSubLabel, stride1);
    } else {
        cout << "Error: lossLayer->prevLayer size does not match label image size." << endl;
        std::exit(EXIT_FAILURE);
    }

    lossLayer->setGroundTruth(*pSubLabel);


    if (nullptr != pRotatedLabel) {
        delete pRotatedLabel;
        pRotatedLabel = nullptr;
    }

    if (nullptr != pSubLabel) {
        delete pSubLabel;
        pSubLabel = nullptr;
    }
}


void HNSegVNet::setInput(const string &filename, const vector<float>& radianVec, const vector<int>& center) {
    InputLayer *inputLayer = getInputLayer();
    Tensor<float>* pImage = nullptr;
    m_pDataMgr->readImageFile(filename, pImage);

    Tensor<float>* pRotatedImage = nullptr;
    pImage->rotate3D(radianVec, IPPI_INTER_CUBIC, pRotatedImage);

    if (nullptr != pImage) {
        delete pImage;
        pImage = nullptr;
    }

    Tensor<float>* pSubImage = new Tensor<float>(inputLayer->m_tensorSize);
    const vector<int> stride1 = vector<int>(inputLayer->m_tensorSize.size(),1);
    const vector<int> topLeft = getTopLeftIndexFrom(pRotatedImage->getDims(), inputLayer->m_tensorSize, center);
    pRotatedImage->subTensorFromTopLeft(topLeft, pSubImage, stride1);
    inputLayer->setInputTensor(*pSubImage);

    if (nullptr != pRotatedImage) {
        delete pRotatedImage;
        pRotatedImage = nullptr;
    }

    if (nullptr != pSubImage) {
        delete pSubImage;
        pSubImage = nullptr;
    }
}


void HNSegVNet::train() {
    DiceLossLayer *lossLayer = (DiceLossLayer *) getFinalLayer();


    m_loss = 0;
    m_dice =0;
    m_TPR = 0;

    int N =m_pDataMgr->m_NTrainFile;
    if (m_OneSampleTrain){
        N = 1;
    }
    const int batchSize = getBatchSize();
    //const float learningRate = getLearningRate();
    const int numBatch = (N + batchSize -1) / batchSize;
    int n = 0;
    int batch = 0;
    vector<int> randSeq = generateRandomSequence(N);

    m_lastBatchLoss = m_batchLoss;
    while (batch < numBatch) {
        zeroParaGradient();
        int i = 0;
        m_batchLoss = 0.0;
        for (i = 0; i < batchSize && n < N; ++i) {
            const string imageFilePath = m_pDataMgr->m_trainImagesVector[randSeq[n]];
            const string labelFilePath = m_pDataMgr->getLabelPathFrom(imageFilePath);

            const vector<float> radianVec = generatePositiveNegativeRandomRadian(3, M_PI/4);
            vector<int>  center;
            setGroundtruth(labelFilePath,radianVec, center, 15);
            setInput(imageFilePath, radianVec, center);

            forwardPropagate();
            m_loss += lossLayer->getLoss();
            m_batchLoss += lossLayer->getLoss();
            // for softmax preceeds over loss layer
            if (m_isSoftmaxBeforeLoss){
                m_dice += lossLayer->diceCoefficient();
                m_TPR  += lossLayer->getTPR();
            }
            else{
                m_dice += lossLayer->diceCoefficient(0.5);
                m_TPR  += lossLayer->getTPR(0.5);
            }

            backwardPropagate(true);
            //debug
            //saveYTensor();
            //savedYTensor();
            ++n;
        }
        averageParaGradient(i);
        m_batchLoss /= i;

        //for parameter-wise learning rates
        float deltaLoss = 0.0;
        if (0.0 != m_lastBatchLoss){
            deltaLoss = m_batchLoss - m_lastBatchLoss;
            updateLearingRates(deltaLoss);
        }
        m_lastBatchLoss = m_batchLoss;
        sgd(i);

        // for global learning rate
        //sgd(learningRate, i)

        ++batch;

    }
    m_loss /=n;
    m_dice /= n;
    m_TPR /= n;



    printf("Train: loss = %f, Dice = %f, TPR = %f; \n", m_loss, m_dice, m_TPR);
}

float HNSegVNet::test() {
    DiceLossLayer *lossLayer = (DiceLossLayer *) getFinalLayer();

    m_loss = 0.0;
    m_dice = 0;
    m_TPR = 0;

    int N = m_pDataMgr->m_NTestFile;
    if (m_OneSampleTrain){
        N = 1;
    }
    int n = 0;
    while (n < N) {
        const string imageFilePath = m_pDataMgr->m_testImagesVector[n];
        const string labelFilePath = m_pDataMgr->getLabelPathFrom(imageFilePath);

        const vector<float> radianVec = {0,0,0};
        vector<int> center;
        setGroundtruth(labelFilePath, radianVec, center, 0);
        setInput(imageFilePath, radianVec, center);

        forwardPropagate();
        m_loss += lossLayer->getLoss();

        //debug
        //cout<<"Image: "<<imageFilePath<<endl;
        //cout<<"Label: "<<labelFilePath<<endl;
        //cout<<"losss = "<<lossLayer->getLoss()<<", center = "<<vector2Str(center)<<endl;
        //cout<<endl;

        // for softmax preceeds over loss layer
        if (m_isSoftmaxBeforeLoss){
            m_dice += lossLayer->diceCoefficient();
            m_TPR  += lossLayer->getTPR();
        }
        else{
            m_dice += lossLayer->diceCoefficient(0.5);
            m_TPR  += lossLayer->getTPR(0.5);
        }

        ++n;

    }
    m_loss /=N;
    m_dice /= N;
    m_TPR /= N;

    printf("Test: loss = %f, Dice = %f, TPR = %f; \n", m_loss, m_dice, m_TPR);

    return  m_loss;

}

float HNSegVNet::test(const string &imageFilePath, const string &labelFilePath) {
    InputLayer *inputLayer = getInputLayer();
    DiceLossLayer *lossLayer = (DiceLossLayer *) getFinalLayer();

    m_loss = 0.0;
    m_dice = 0.0;
    m_TPR = 0.0;

    vector<int> center;
    setGroundtruth(labelFilePath, {0,0,0}, center, 0);
    setInput(imageFilePath, {0,0,0}, center);

    forwardPropagate();

    // output float image for debug
    //if (!m_isSoftmaxBeforeLoss) {
    //    const string floatImageOutput = m_pDataMgr->generateFloatImagePath(imageFilePath);
    //    m_pDataMgr->saveImage2File(lossLayer->m_prevLayer->m_pYTensor, offset, floatImageOutput);
    // }

    //Output network predicted label
    string outputLabelFilePath = m_pDataMgr->generateLabelFilePath(imageFilePath);
    vector<int> offset = m_pDataMgr->getOutputOffset(inputLayer->m_tensorSize, center);
    if (m_isSoftmaxBeforeLoss){
        m_pDataMgr->saveOneHotCode2LabelFile(lossLayer->m_prevLayer->m_pYTensor, outputLabelFilePath, offset);
    }
    else{
        Tensor<unsigned char> predictResult(lossLayer->m_prevLayer->m_tensorSize);
        lossLayer->getPredictTensor(predictResult, 0.5);
        m_pDataMgr->saveLabel2File(&predictResult, offset, outputLabelFilePath);
    }

    if (!labelFilePath.empty()){
        m_loss = lossLayer->getLoss();
        if (m_isSoftmaxBeforeLoss){
            m_dice = lossLayer->diceCoefficient();
            m_TPR  = lossLayer->getTPR();
        }
        else{
            m_dice = lossLayer->diceCoefficient(0.5);
            m_TPR  = lossLayer->getTPR(0.5);
        }
        printf("Test: loss = %f, Dice = %f, TPR = %f; \n", m_loss, m_dice, m_TPR);
    }
    return m_loss;
}

void HNSegVNet::detectSoftmaxBeforeLoss() {
    m_isSoftmaxBeforeLoss = true;
    if ("SigmoidLayer" == getFinalLayer()->m_prevLayer->m_type){
        m_isSoftmaxBeforeLoss = false;
    }
}

