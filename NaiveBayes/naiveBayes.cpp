#include "naive_bayes.h"
#include<iostream>
// #include<wchar.h>



void NaiveBayes::AddData(std::string cl,std::vector<int> dataF){

    data d;
    d.label = NaiveBayes::labelmap[cl];
    d.val = dataF;
    NaiveBayes::trainData.push_back(d);
}

void NaiveBayes::calculateProbs(){

    

    for (int i=0; i < trainData.size();i++){

        classProbMap[trainData[i].label]++;
       
        
        for (int j=0;j< NaiveBayes::featureVecsize; j++){
             if (trainData[i].val[j] !=0) {
                
                if (classProbFeaturewise[trainData[i].label].size() == 0){
                       std::vector<double> featCounter(featureVecsize+1,0);
                       classProbFeaturewise[trainData[i].label] = featCounter  ;

                }
             classProbFeaturewise[trainData[i].label][j]++;}
        }
    }
    std::map<int, double>::iterator it = classProbMap.begin();
    
    while (it!=classProbMap.end()){
        it->second = it->second/trainData.size();
        std::vector<double>::iterator it_feat = classProbFeaturewise[it->first].begin();
            int SumFeatureClass = 0 ;
            while (it_feat!= classProbFeaturewise[it->first].end()){

                SumFeatureClass = SumFeatureClass + *it_feat;
                it_feat++;
            }
            it_feat = classProbFeaturewise[it->first].begin();
            while (it_feat!= classProbFeaturewise[it->first].end()){

                *it_feat = *it_feat/SumFeatureClass;
                it_feat++;
            }
        it++;
        
    }

}
void NaiveBayes::train(){
    NaiveBayes::featureVecsize = trainData[0].val.size();
    calculateProbs();

}
std::vector<double> NaiveBayes::Predict(std::vector<int> da){

    std::vector<double> probVec;
    double p;
    for (int x = 0; x < labelmap.size();x++){
        p = classProbMap[x];
        for (int j = 0 ; j < da.size(); j++){
            if (da[j] !=0 ){
                p = p * classProbFeaturewise[x][j];
            }
        
        }
        probVec.push_back(p);
    }
    return probVec;

}


int main (){


    NaiveBayes nb;
    std::cout<<"jjjjjjjjj"<<"\n";
    nb.AddData("first",{0,0,1,1,0,0,1});
    nb.AddData("second",{0,0,1,1,0,0,1});
    nb.AddData("third",{0,0,1,1,0,0,1});
    nb.AddData("fourth",{0,0,1,1,1,0,1});
    nb.AddData("third",{0,0,1,1,0,0,1});
    nb.AddData("fourth",{1,0,1,1,0,0,1});
    nb.AddData("first",{1,0,1,1,0,0,1});
    nb.AddData("second",{0,0,1,0,1,1,1});
    nb.AddData("fourth",{0,0,1,0,1,1,1});
    nb.AddData("second",{0,0,1,0,1,1,1});
    nb.AddData("third",{1,0,1,0,1,1,1});
    nb.AddData("first",{1,1,1,0,1,1,1});
    nb.train();
    std::vector<double> res =  nb.Predict({0,0,0,0,1,0,1});
    std::vector<double>::iterator it = res.begin();
    while (it!=res.end()){
        std::cout<< *it<<"\n";
        it ++;

    }

   return 0;

}


