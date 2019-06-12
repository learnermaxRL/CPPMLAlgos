/* Given a finite set of labels the algorithm assigns labels to some input data,this algo assumes that   */
#include <map>
#include<vector>
#include<string>




struct data{

    int label;
    std::vector<int> val;


};

class NaiveBayes {

    private:
        int numClasses;
        std::vector<data> trainData;
        int featureVecsize;
        std::map<std::string,int> labelmap ;
        std::map<int,double> classProbMap;
        std::map<int,std::vector<double> > classProbFeaturewise;

    
    public:
        NaiveBayes(){

    labelmap.insert({"first",0});
    labelmap.insert({"second",1});
    labelmap.insert({"third",2});
    // std::cout<<"done..!"<<"\n";

};
         
         void calculateProbs();
         void train();
         std::vector<double>  Predict(std::vector<int>);
         void AddData(std::string ,std::vector<int> );
};