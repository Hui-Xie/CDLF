
#include <stdio.h>
#include <string>
#include <vector>
#include <Tensor.h>

using namespace std;


//TCIACode, Age, Survival(months),AliveorDead,CauseofDeath

struct Survival{
    string m_patientCode;
    int m_age;
    float  m_survivalMonth;
    bool m_isAlive;
    string m_causeOfDeath;
};


class HNClinicalDataMgr {
public:
    HNClinicalDataMgr();
    ~HNClinicalDataMgr();

    void readSurvivalData(const string & filename);

    // get a 2*10 tensor of 10 years (dead(0), alive(1)) probability
    Tensor<float>  getSurvivalTensor(struct Survival&  survival);

    vector<struct Survival> m_survivalVector;

};