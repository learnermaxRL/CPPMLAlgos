#include "naive_bayes.h"
#include<iostream>



// this function adds data to our training list
void NaiveBayes::AddData(std::string cl,std::vector<int> dataF){

    //check if the label is  part of label map ,add otherwise
    if (labelmap.find(cl) == labelmap.end()){
        labelmap.insert ({cl,numClasses});
        numClasses ++;
    }
    //each data is pushed into vector as label which is int,and feature vector as val in struct
    //TO:DO add error hadnling in case feature vector size varies
    data d;
    d.label = NaiveBayes::labelmap[cl];
    d.val = dataF;
    NaiveBayes::trainData.push_back(d);
}

void NaiveBayes::calculateProbs(){

 // Initialzing map which will store conditional probability of feature vec wrt classes (P(xi/C))
    for (int labLength = 0; labLength < labelmap.size();labLength++){
        
        std::vector<double> featCounter(featureVecsize+1,0);
        classProbFeaturewise[labLength] = featCounter;

    }

// Iterating through train data 
    for (int i=0; i < trainData.size();i++){

        //calculating class occurences
        classProbMap[trainData[i].label]++;
       
        //calculating each feature occurence per class
        for (int j=0;j< NaiveBayes::featureVecsize; j++){
             if (trainData[i].val[j] !=0) {
                
             classProbFeaturewise[trainData[i].label][j]++;}
        }
    }
    std::map<int, double>::iterator it = classProbMap.begin();
    
// converting counters to probabailities
    while (it!=classProbMap.end()){
        it->second = it->second/trainData.size(); //converting class counter to probability
        
        std::vector<double>::iterator it_feat = classProbFeaturewise[it->first].begin();
            int SumFeatureClass = 0 ;
            //calculating total num of counts of each feature
            while (it_feat!= classProbFeaturewise[it->first].end()){

                SumFeatureClass = SumFeatureClass + *it_feat;
                it_feat++;
            }
            //converting each feature counter per class to probability
            it_feat = classProbFeaturewise[it->first].begin();
            while (it_feat!= classProbFeaturewise[it->first].end()){

                *it_feat = *it_feat/SumFeatureClass;
                it_feat++;
            }
        it++;
        
    }

}
void NaiveBayes::train(){
    //initializing feature vector size TO:DO add error handling
    NaiveBayes::featureVecsize = trainData[0].val.size();
    calculateProbs();

}
std::vector<double> NaiveBayes::Predict(std::vector<int> da){

    std::string predicClass; //variable which will store class with highest probabaility
    double highest = 0; //variable which will store prob of class with highest probabaility
    std::cout << "Predicting.."<<'\n';
    std::vector<double> probVec;
    double p;
    std::map<std::string,int>::iterator iteratormap = labelmap.begin();
    while (iteratormap != labelmap.end()) {
        p = classProbMap[iteratormap->second]; //(P(Ck))
        for (int j = 0 ; j < da.size(); j++){ //looping through each feature in feature vector and checking if its 0 or 1
            if (da[j] !=0 ){
                //product of (P(xi/C))
                p = p * classProbFeaturewise[iteratormap->second][j];
            }
        
        }
        // std::cout<< p << " "<< iteratormap->first<<"\n";
        if (p>highest){
            // if probability of class is highest than current assign that as highest probability
            predicClass = iteratormap->first;
            highest=p;
        }
        iteratormap++;
    }
    std::cout <<"Predicted class is "<<predicClass<<std::endl;
    return probVec;

}


int main (){


    NaiveBayes nb;
 // add some random data

    nb.AddData("first",{0,0,1,1,0,0,1});
    nb.AddData("first",{0,0,1,1,0,0,1});
    nb.AddData("first",{0,0,1,1,0,0,1});
    nb.AddData("fifth",{0,0,1,1,1,0,1});
    nb.AddData("third",{0,0,1,1,0,0,1});
    nb.AddData("fourth",{1,0,1,1,0,0,1});
    nb.AddData("first",{1,0,1,1,0,0,1});
    nb.AddData("first",{0,0,1,0,1,1,1});
    nb.AddData("fourth",{0,0,1,0,1,1,1});
    nb.AddData("first",{0,0,1,0,1,1,1});
    nb.AddData("third",{1,0,1,0,1,1,1});
    nb.AddData("first",{1,1,1,0,1,1,1});
    // train NB algo
    nb.train();

    //get prediction for new point
    std::vector<double> res =  nb.Predict({0,0,0,0,1,0,1});
    

   return 0;

}


