//
// Created by Hui Xie on 9/15/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "Net.h"
#include "GPUAttr.h"
#include "FileTools.h"
#include <cstdio>
#include <unistd.h>
#include <fstream>
#include "Tools.h"
#include "AllLayers.h"

Net::Net(const string &name, const string& saveDir) {
    m_name = eraseAllSpaces(name);
    m_layers.clear();
    m_learningRate = 0.001;
    m_lossTolerance = 0.02;
    m_judgeLoss = true;
    m_batchSize = 1;
    m_epoch = 0;
    setDir(saveDir);
    m_unlearningLayerID = 0;
}

Net::~Net() {
    for (map<int, Layer *>::iterator it = m_layers.begin(); it != m_layers.end(); ++it) {
        if (nullptr != it->second) {
            delete it->second;
            it->second = nullptr;
        }
    }
    m_layers.clear();
}

void Net::setLearningRate(const float learningRate) {
    m_learningRate = learningRate;
}

void Net::setLossTolerance(const float tolerance) {
    m_lossTolerance = tolerance;
}

void Net::setJudgeLoss(const bool judgeLoss) {
    m_judgeLoss = judgeLoss;
}

void Net::setBatchSize(const int batchSize) {
    m_batchSize = batchSize;
}

void Net::setEpoch(const int epoch) {
    m_epoch = epoch;
}

void Net::setDir(const string dir) {
    string netDir = dir;
    if ("." == netDir) {
        char cwd[PATH_MAX];
        char* result = getcwd(cwd, sizeof(cwd));
        if (nullptr == result){
            cout<<"Error: program can not correctly get current working directory. Promram only support Linux."<<endl;
            return;
        }
        netDir = string(cwd);
    }
    netDir += "/" + m_name;
    createDir(netDir);
    m_directory = netDir;
}

void Net::setUnlearningLayerID(const int id){
    m_unlearningLayerID = id;
}

string Net::getName() {
    return m_name;
}

float Net::getLearningRate() {
    return m_learningRate;
}

float Net::getLossTolerance() {
    return m_lossTolerance;
}

bool Net::getJudgeLoss() {
    return m_judgeLoss;
}

int Net::getBatchSize() {
    return m_batchSize;
}

int Net::getEpoch() {
    return m_epoch;
}

string Net::getDir() {
    return m_directory;
}

int Net::getUnlearningLayerID(){
    return m_unlearningLayerID;
}

map<int, Layer *> Net::getLayersMap() {
    return m_layers;
}

int Net::getNumParameters() {
    int num = 0;
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        if (iter->second->m_id > m_unlearningLayerID){
            num += iter->second->getNumParameters();
        }
    }
    return num;
}

void Net::zeroParaGradient() {
    for (map<int, Layer *>::reverse_iterator rit = m_layers.rbegin(); rit != m_layers.rend(); ++rit) {
        rit->second->zeroParaGradient();
    }
}

void Net::addLayer(Layer *layer) {
    if (nullptr == layer) return;
    if (layer->m_id <= 0){
        cout<<"Error: layerID = 0 reserves for nonexist layer."<<endl;
        return;
    }

    if (0 == m_layers.count(layer->m_id) && !layerExist(layer)) {
        m_layers[layer->m_id] = layer;
    } else {
        cout << "Error: repeated layer ID or name in added layer." << endl;
        cout << "\t layer ID: " << layer->m_id << endl;
        cout << "\t already existed layer: " << m_layers[layer->m_id]->m_name << endl;
        cout << "\t new adding layer: " << layer->m_name << endl;
    }
}

Layer* Net::getLayer(const int ID) {
    return m_layers.at(ID);
}

vector<Layer*> Net::getLayers(const vector<int> IDVec){
    vector<Layer*> pLayersVec;
    const int N = IDVec.size();
    for (int i=0; i<N; ++i){
        pLayersVec.push_back(m_layers.at(IDVec[i]));
    }
    return pLayersVec;
}


InputLayer *Net::getInputLayer() {
    return (InputLayer *) m_layers.begin()->second;
}

Layer *Net::getFirstLayer() {
    return m_layers.begin()->second;
}

Layer *Net::getFinalLayer() {
    return m_layers.rbegin()->second;
}

void Net::initialize() {
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        iter->second->initialize("Xavier");
    }
}

void Net::printIteration(LossLayer *lossLayer, const int nIter, const bool transpose) {
    cout << "Iteration: " << nIter << "  " << "Output Result: " << endl;
    if (transpose){
        lossLayer->m_prevLayer->m_pYTensor->transpose().print();
    }
    else{
        lossLayer->m_prevLayer->m_pYTensor->print();
    }
    if (nullptr != lossLayer->m_pGroundTruth) {
        cout << "GrounTruth: " << endl;
        if (transpose){
            lossLayer->m_pGroundTruth->transpose().print();
        }
        else{
            lossLayer->m_pGroundTruth->print();
        }

    }
    cout << "Loss: " << lossLayer->lossCompute() << endl;
    cout << endl;
}

void Net::printLayersY() {
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        iter->second->printY();
    }
}

void Net::printLayersDY() {
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        iter->second->printDY();
    }
}


void Net::printArchitecture() {
    cout << endl << "========== Network Architecture of " << m_name << " =============" << endl;
    cout << "===========================================================" << endl;
    for (map<int, Layer *>::const_iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        iter->second->printStruct();
    }
    cout << "Total learning parameters: " << getNumParameters() << endl;
    cout << "=========== End of Network Architecture ==================" << endl;
}

bool Net::layerExist(const Layer *layer) {
    for (map<int, Layer *>::const_iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        if (layer->m_name == iter->second->m_name || layer == iter->second) {
            cout << "Error: " << layer->m_name << " has already been in the previous added layer." << endl;
            return true;
        }
    }
    return false;
}

void Net::saveLayersStruct() {
    const string tableHead = "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/Lambda, StartPosition, \r\n";
    string filename = m_directory + "/LayersStruct.csv";
    FILE *pFile;
    pFile = fopen(filename.c_str(), "w");
    if (nullptr == pFile) {
        printf("Error: can not open  %s  file.\n", filename.c_str());
        return;
    }
    fputs(tableHead.c_str(), pFile);
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        iter->second->saveStructLine(pFile);
    }
    fclose(pFile);
}

void Net::readLayesStruct(vector<struct LayerStruct> &layersStructVec) {
    string filename = m_directory + "/LayersStruct.csv";
    ifstream ifs(filename.c_str());;
    char lineChar[120];
    char type[30];
    char name[30];
    char outputTensorSizeChar[30];
    char filterSizeChar[30];
    char strideChar[30];
    char startPosition[30];
    char preLayersIDsChar[30];
    if (ifs.good()) {
        ifs.ignore(120, '\n'); // ignore the table head
        while (ifs.peek() != EOF) {
            ifs.getline(lineChar, 120, '\n');
            for (int i = 0; i < 120; ++i) {
                if (lineChar[i] == ',') lineChar[i] = ' ';
            }
            // tableHead: ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition,
            struct LayerStruct layerStruct;
            int nFills = sscanf(lineChar, "%d  %s  %s  %s  %s  %s  %s  %d  %f  %s  \r\n",
                                &layerStruct.m_id, type, name, preLayersIDsChar, outputTensorSizeChar,
                                filterSizeChar, strideChar, &layerStruct.m_numFilter,  &layerStruct.m_k, startPosition);
            if (9 != nFills) {
                cout << "Error: sscanf netParameterChar in loadLayersStruct at line: " << string(lineChar) << endl;
            } else {
                layerStruct.m_type = string(type);
                layerStruct.m_name = string(name);
                layerStruct.m_preLayersIDs = str2Vector(string(preLayersIDsChar));
                if (0 < layerStruct.m_preLayersIDs.size()){
                    layerStruct.m_preLayerID = layerStruct.m_preLayersIDs[0];
                }
                layerStruct.m_outputTensorSize = str2Vector(string(outputTensorSizeChar));
                layerStruct.m_filterSize = str2Vector(string(filterSizeChar));
                layerStruct.m_stride = str2Vector(string(strideChar));
                layerStruct.m_startPosition = str2Vector(string(startPosition));

                layersStructVec.push_back(layerStruct);
            }
        }
    }
    ifs.close();
}


void Net::createLayers(const vector<struct LayerStruct> &layersStructVec) {
    int N = layersStructVec.size();
    if (0 == N){
        cout<<"Error: layer struct vector is empty."<<endl;
        return;
    }
    else{
        cout<<"Info: program loaded "<<N <<" layers"<<endl;
    }

    Layer *pLayer = nullptr;
    Layer *pPreLayer = nullptr;
    for (int i = 0; i < N; ++i) {
        const struct LayerStruct &s = layersStructVec[i];
        if (0 != s.m_preLayerID){
            pPreLayer = m_layers[s.m_preLayerID];
        }

        if ("InputLayer" == s.m_type) {
            pLayer = new InputLayer(s.m_id, s.m_name, s.m_outputTensorSize);
        }
        else if ("FCLayer" == s.m_type) {
            pLayer = new FCLayer(s.m_id, s.m_name, pPreLayer, s.m_outputTensorSize[0]);
        }
        else if ("ReLU" == s.m_type) {
            pLayer = new ReLU(s.m_id, s.m_name, pPreLayer, s.m_outputTensorSize, s.m_k);
        }
        else if ("NormalizationLayer" == s.m_type) {
            pLayer = new NormalizationLayer(s.m_id, s.m_name, pPreLayer, s.m_outputTensorSize);
        }
        else if ("SoftmaxLayer" == s.m_type) {
            pLayer = new SoftmaxLayer(s.m_id, s.m_name, pPreLayer);
        }
        else if ("ConvolutionLayer" == s.m_type) {
           pLayer = new ConvolutionLayer(s.m_id, s.m_name, pPreLayer, s.m_filterSize, s.m_stride, s.m_numFilter);
        }
        else if ("LeftMatrixLayer" == s.m_type) {
            pLayer = new LeftMatrixLayer(s.m_id, s.m_name, pPreLayer, s.m_filterSize);
        }
        else if ("RightMatrixLayer" == s.m_type) {
            pLayer = new RightMatrixLayer(s.m_id, s.m_name, pPreLayer, s.m_filterSize);
        }
        else if ("TransposedConvolutionLayer" == s.m_type) {
            pLayer = new TransposedConvolutionLayer(s.m_id, s.m_name, pPreLayer, s.m_filterSize, s.m_stride, s.m_numFilter);
        }
        else if ("BranchLayer" == s.m_type) {
           pLayer = new BranchLayer(s.m_id, s.m_name, pPreLayer);
        }
        else if ("LinearLayer" == s.m_type) {
           pLayer = new LinearLayer(s.m_id, s.m_name, pPreLayer);
        }
        else if ("CrossEntropyLoss" == s.m_type) {
           pLayer = new CrossEntropyLoss(s.m_id, s.m_name, pPreLayer);
        }
        else if ("SquareLossLayer" == s.m_type) {
            pLayer = new SquareLossLayer(s.m_id, s.m_name, pPreLayer, s.m_k);
        }
        else if ("DiceLossLayer" == s.m_type) {
            pLayer = new DiceLossLayer(s.m_id, s.m_name, pPreLayer);
        }
        else if ("ExponentialLayer" == s.m_type) {
           pLayer = new ExponentialLayer(s.m_id, s.m_name, pPreLayer);
        }
        else if ("IdentityLayer" == s.m_type) {
           pLayer = new IdentityLayer(s.m_id, s.m_name, pPreLayer);
        }
        else if ("MaxPoolingLayer" == s.m_type) {
           pLayer = new MaxPoolingLayer(s.m_id, s.m_name, pPreLayer, s.m_filterSize, s.m_stride);
        }
        else if ("MergerLayer" == s.m_type) {
           pLayer = new MergerLayer(s.m_id, s.m_name, s.m_outputTensorSize);
           int nInBranches = s.m_preLayersIDs.size();
           for(int k=0; k<nInBranches; ++k){
               pLayer->addPreviousLayer(m_layers[s.m_preLayersIDs[k]]);
           }
        }
        else if ("ConcatenateLayer" == s.m_type) {
            vector<Layer*> pLayersVec = getLayers(s.m_preLayersIDs);
            pLayer = new ConcatenateLayer(s.m_id, s.m_name, pLayersVec, s.m_outputTensorSize);
        }
        else if ("SigmoidLayer" == s.m_type) {
           pLayer = new SigmoidLayer(s.m_id, s.m_name, pPreLayer, s.m_outputTensorSize, (int)s.m_k);
        }
        else if ("RescaleLayer" == s.m_type) {
            pLayer = new RescaleLayer(s.m_id, s.m_name, pPreLayer, (int)s.m_k);
        }
        else if ("SubTensorLayer" == s.m_type) {
           pLayer = new SubTensorLayer(s.m_id, s.m_name, pPreLayer, s.m_startPosition, s.m_outputTensorSize);
        }
        else if ("PaddingLayer" == s.m_type) {
            pLayer = new PaddingLayer(s.m_id, s.m_name, pPreLayer, s.m_outputTensorSize);
        }
        else if ("VectorizationLayer" == s.m_type) {
           pLayer = new VectorizationLayer(s.m_id, s.m_name, pPreLayer);
        }
        else if ("ReshapeLayer" == s.m_type) {
            pLayer = new ReshapeLayer(s.m_id, s.m_name, pPreLayer, s.m_outputTensorSize);
        }
        else if ("AssemblyLossLayer" == s.m_type) {
            pLayer = new AssemblyLossLayer(s.m_id, s.m_name, pPreLayer);
        }
        else if ("LossConvexExample2" == s.m_type) {
           pLayer = new LossConvexExample2(s.m_id, s.m_name, pPreLayer);
        }
        else if ("LossNonConvexExample1" == s.m_type) {
           pLayer = new LossNonConvexExample1(s.m_id, s.m_name, pPreLayer);
        }
        else if ("LossNonConvexExample2" == s.m_type) {
           pLayer = new LossNonConvexExample2(s.m_id, s.m_name, pPreLayer);
        }
        else if ("LossConvexExample1" == s.m_type) {
           pLayer = new LossConvexExample1(s.m_id, s.m_name, pPreLayer);
        }
        else {
            cout << "Error: " << layersStructVec[i].m_type << " does not support at the " << i << " line." << endl;
            break;
        }
        addLayer(pLayer);

    }
}


// tableHead: ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, StartPosition,
//load layers structure file and create layers
void Net::loadLayersStruct() {
    vector<struct LayerStruct> layersStructVec;
    readLayesStruct(layersStructVec);
    createLayers(layersStructVec);
}

void Net::saveLayersParameters() {
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        const string layerDir = m_directory + "/" + to_string(iter->second->m_id);
        if (iter->second->m_id < m_unlearningLayerID && dirExist(layerDir)){
            continue;
        }
        else{
            iter->second->save(m_directory);
        }
    }
}

void Net::loadLayersParameters() {
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        iter->second->load(m_directory);
    }
}

void Net::saveNetParameters() {
    const string tableHead = "Name, LearningRate, BatchSize, Epoch, LossTolerance, JudgeLoss, UnlearningID, \r\n";
    string filename = m_directory + "/NetParameters.csv";
    FILE *pFile;
    pFile = fopen(filename.c_str(), "w");
    if (nullptr == pFile) {
        printf("Error: can not open  %s  file for writing.\n", filename.c_str());
        return;
    }
    fputs(tableHead.c_str(), pFile);
    fprintf(pFile, "%s, %f, %d, %d, %f, %d, %d, \r\n", m_name.c_str(), m_learningRate, m_batchSize, m_epoch,
            m_lossTolerance, m_judgeLoss ? 1 : 0, m_unlearningLayerID);
    fclose(pFile);
}

void Net::loadNetParameters() {
    string filename = m_directory + "/NetParameters.csv";
    ifstream ifs(filename.c_str());;
    char netParameterChar[200];
    int judgeLoss = 0;
    char name[100];
    if (ifs.good()) {
        ifs.ignore(200, '\n'); // ignore the table head
        ifs.getline(netParameterChar, 200, '\n');
        for (int i = 0; i < 100; ++i) {
            if (netParameterChar[i] == ',') netParameterChar[i] = ' ';
        }
        int nFills = sscanf(netParameterChar, "%s  %f  %d  %d  %f  %d  %d \r\n", name, &m_learningRate, &m_batchSize,
                            &m_epoch, &m_lossTolerance, &judgeLoss, &m_unlearningLayerID);
        if (7 != nFills) {
            cout << "Error: sscanf netParameterChar in loadNetParameters." << endl;
        }
    }
    m_name = string(name);
    m_judgeLoss = (judgeLoss == 1) ? true : false;
    ifs.close();
}

void Net::save() {
    cout<<"Net parameters start to save ....."<<endl;
    saveLayersStruct();
    saveNetParameters();
    saveLayersParameters();
    cout << "Net parameters saved at " << m_directory << " directory." << endl;
}

void Net::saveYTensor(){
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        const string fileName = m_directory +"/Y_"+ to_string(iter->second->m_id) + ".csv";
        if (nullptr != iter->second->m_pYTensor ){
            iter->second->m_pYTensor->save(fileName,false);
        }
    }
}

void Net::savedYTensor(){
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        const string fileName = m_directory +"/dY_"+ to_string(iter->second->m_id) + ".csv";
        if (nullptr != iter->second->m_pdYTensor ){
            iter->second->m_pdYTensor->save(fileName,false);
        }
    }
}

void Net::load() {
    loadLayersStruct();
    loadNetParameters();
    loadLayersParameters();
    cout<<"Net parameters were loaded from "<<m_directory<<endl;
}
