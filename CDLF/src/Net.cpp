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

void Net::setEpoch(const long epoch) {
    m_epoch = epoch;
}

void Net::setDir(const string dir) {
    string netDir = dir;
    if ("." == netDir) {
        char cwd[PATH_MAX];
        getcwd(cwd, sizeof(cwd));
        netDir = string(cwd);
    }
    netDir += "/" + m_name;
    createDir(netDir);
    m_directory = netDir;
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

long Net::getEpoch() {
    return m_epoch;
}

string Net::getDir() {
    return m_directory;
}

map<int, Layer *> Net::getLayersMap() {
    return m_layers;
}

long Net::getNumParameters() {
    long num = 0;
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        num += iter->second->getNumParameters();
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

Layer *Net::getLayer(const int ID) {
    return m_layers.at(ID);
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

void Net::printIteration(LossLayer *lossLayer, const int nIter) {
    cout << "Iteration: " << nIter << "  " << "Output Result: " << endl;
    long N = lossLayer->m_prevLayer->m_pYTensor->getLength();
    lossLayer->m_prevLayer->m_pYTensor->reshape({1, N}).printElements();
    if (nullptr != lossLayer->m_pGroundTruth) {
        cout << "GrounTruth: " << endl;
        lossLayer->m_pGroundTruth->reshape({1, N}).printElements();
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
    cout << "======================================================" << endl;
    int i = 1;
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        cout << "Layer_" << i++ << " (" << iter->second->m_type << ", id=" << std::to_string(iter->second->m_id)
             << "): " << iter->second->m_name << " : ";
        if (nullptr != iter->second->m_prevLayer) {
            cout << "PreviousLayer = " << iter->second->m_prevLayer->m_name << "; ";
        }
        if ("ConvolutionLayer" == iter->second->m_type) {
            cout << "FilterSize = " << vector2Str(((ConvolutionLayer *) iter->second)->m_filterSize) << "; "
                 << "NumOfFilter = " << ((ConvolutionLayer *) iter->second)->m_numFilters << "; ";
        }
        if ("MaxPoolingLayer" == iter->second->m_type) {
            cout << "FilterSize = " << vector2Str(((MaxPoolingLayer *) iter->second)->m_filterSize) << "; ";
        }
        cout << "OutputSize = " << vector2Str(iter->second->m_tensorSize) << "; " << endl;
    }

    cout << "This network has total " << getNumParameters() << " learning parameters. " << endl;
    cout << "=========== End of Network Architecture ==============" << endl;
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
    const string tableHead = "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n";
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
    char startPosition[30];
    char preLayersIDsChar[30];
    if (ifs.good()) {
        ifs.ignore(120, '\n'); // ignore the table head
        while (ifs.peek() != EOF) {
            ifs.getline(lineChar, 120, '\n');
            for (int i = 0; i < 120; ++i) {
                if (lineChar[i] == ',') lineChar[i] = ' ';
            }
            // tableHead: ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride, StartPosition,
            struct LayerStruct layerStruct;
            int nFills = sscanf(lineChar, "%d  %s  %s  %s  %s  %s  %d  %d  %s  \r\n",
                                &layerStruct.m_id, type, name, preLayersIDsChar, outputTensorSizeChar,
                                filterSizeChar, &layerStruct.m_numFilter, &layerStruct.m_stride, startPosition);
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
                layerStruct.m_startPosition = str2Vector(string(startPosition));

                layersStructVec.push_back(layerStruct);
            }
        }
    }
    ifs.close();
}


void Net::createLayers(const vector<struct LayerStruct> &layersStructVec) {
    int N = layersStructVec.size();
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
            pLayer = new ReLU(s.m_id, s.m_name, pPreLayer);
        }
        else if ("NormalizationLayer" == s.m_type) {
            pLayer = new NormalizationLayer(s.m_id, s.m_name, pPreLayer);
        }
        else if ("SoftmaxLayer" == s.m_type) {
            pLayer = new SoftmaxLayer(s.m_id, s.m_name, pPreLayer);
        }
        else if ("ConvolutionLayer" == s.m_type) {
           pLayer = new ConvolutionLayer(s.m_id, s.m_name, pPreLayer, s.m_filterSize, s.m_numFilter, s.m_stride);
        }
        else if ("BranchLayer" == s.m_type) {
           pLayer = new BranchLayer(s.m_id, s.m_name, pPreLayer);
        }
        else if ("BiasLayer" == s.m_type) {
           pLayer = new BiasLayer(s.m_id, s.m_name, pPreLayer);
        }
        else if ("CrossEntropyLoss" == s.m_type) {
           pLayer = new CrossEntropyLoss(s.m_id, s.m_name, pPreLayer);
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
        else if ("ScaleLayer" == s.m_type) {
           pLayer = new ScaleLayer(s.m_id, s.m_name, pPreLayer);
        }
        else if ("SigmoidLayer" == s.m_type) {
           pLayer = new SigmoidLayer(s.m_id, s.m_name, pPreLayer, s.m_stride);
        }
        else if ("SubTensorLayer" == s.m_type) {
           pLayer = new SubTensorLayer(s.m_id, s.m_name, pPreLayer, s.m_startPosition, s.m_outputTensorSize);
        }
        else if ("VectorizationLayer" == s.m_type) {
           pLayer = new VectorizationLayer(s.m_id, s.m_name, pPreLayer);
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
        iter->second->save(m_directory);
    }
}

void Net::loadLayersParameters() {
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        iter->second->load(m_directory);
    }
}

void Net::saveNetParameters() {
    const string tableHead = "Name, LearningRate, BatchSize, Epoch, LossTolerance, JudgeLoss, \r\n";
    string filename = m_directory + "/NetParameters.csv";
    FILE *pFile;
    pFile = fopen(filename.c_str(), "w");
    if (nullptr == pFile) {
        printf("Error: can not open  %s  file for writing.\n", filename.c_str());
        return;
    }
    fputs(tableHead.c_str(), pFile);
    fprintf(pFile, "%s, %f, %d, %ld, %f, %d, \r\n", m_name.c_str(), m_learningRate, m_batchSize, m_epoch,
            m_lossTolerance, m_judgeLoss ? 1 : 0);
    fclose(pFile);
}

void Net::loadNetParameters() {
    string filename = m_directory + "/NetParameters.csv";
    ifstream ifs(filename.c_str());;
    char netParameterChar[100];
    int judgeLoss = 0;
    char name[100];
    if (ifs.good()) {
        ifs.ignore(100, '\n'); // ignore the table head
        ifs.getline(netParameterChar, 100, '\n');
        for (int i = 0; i < 100; ++i) {
            if (netParameterChar[i] == ',') netParameterChar[i] = ' ';
        }
        int nFills = sscanf(netParameterChar, "%s  %f  %d  %ld  %f  %d  \r\n", name, &m_learningRate, &m_batchSize,
                            &m_epoch, &m_lossTolerance, &judgeLoss);
        if (6 != nFills) {
            cout << "Error: sscanf netParameterChar in loadNetParameters." << endl;
        }
    }
    m_name = string(name);
    m_judgeLoss = judgeLoss == 1 ? true : false;
    ifs.close();
}

void Net::save() {
    saveLayersStruct();
    saveNetParameters();
    saveLayersParameters();
    cout << "Net parameters saved at " << m_directory << " directory." << endl;
}

void Net::load() {
    loadLayersStruct();
    loadNetParameters();
    loadLayersParameters();
    cout<<"Net parameters were loaded from "<<m_directory<<endl;
}
