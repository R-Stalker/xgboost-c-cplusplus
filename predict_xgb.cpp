#include <iostream>
#include <fstream>
#include <stdint.h>
#include <string>
#include <math.h>
#include "xgboost/c_api.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::ios;
using std::string;


DMatrixHandle load_dense_data(const char *data_path) {
    ifstream infile;//定义读取文件流，相对于程序来说是in
//    int num_row = 0;
//    string temp;
    infile.open(data_path);//打开文件
    if (!infile) {
        cout << "can not open file" << endl;
    }
//    else {
//        while (getline(infile, temp, '\n')) {
//            num_row++;
//        }
//    }
//    cout << "num_row:" << num_row << endl;
    int num_row = 1000;
    const int feature_count = 108;
    // float **feat = new float *[num_row];
    // for (int i = 0; i < num_row; i++) {
    //     feat[i] = new float[feature_count + 1];
    // }
    //
    float feat[num_row][feature_count + 1];
    for (int i = 0; i < num_row; i++) {
        //xgboost训练和预测使用的数据格式必须保持一致,否则预测结果会异常
        //比如model下的LocalFile模型是用libsvm训练的，预测时候使用的数据必须使用libsvm格式
        feat[i][0] = -1.0; //但我这里是dense格式的数据，为了和libsvim保持一致，则只需在特征列前加一维度即可，可以是任意数字
        for (int j = 1; j <= feature_count; j++) {
            infile >> feat[i][j];//读取一个值（空格、制表符、换行隔开）就写入到矩阵中，行列不断循环进行
        }
    }
    infile.close();//读取完成之后关闭文件

    cout << feat[0][0] << " " << float(feat[0][1]) << " " << float(feat[0][108]) << endl;
    cout << feat[1][0] << " " << float(feat[1][1]) << " " << float(feat[1][109]) << endl;
    cout << feat[2][0] << " " << float(feat[2][1]) << " " << float(feat[2][108]) << endl;
    // convert 2d array to DMatrix
    DMatrixHandle data_DMatrix;
    XGDMatrixCreateFromMat(reinterpret_cast<float *>(feat), (bst_ulong) num_row, (bst_ulong) (feature_count + 1), NAN,
                           &data_DMatrix);

    return data_DMatrix;
}


int main(int argc, char const *argv[]) {
    const char *model_path = "model_file";
    const char *test_data_path = "dense.features";
    const char *predict_data_save_path = "result.txt";

    // create booster handle first
    BoosterHandle booster;
    XGBoosterCreate(NULL, 0, &booster);
    // load model
    XGBoosterLoadModel(booster, model_path);

    // load data
    DMatrixHandle test_data;
    test_data = load_dense_data(test_data_path);

    // predict
    bst_ulong out_len;
    const float *predict_result;
    XGBoosterPredict(booster, test_data, 0, 0, &out_len, &predict_result);

    // save the predict result
    ofstream outfile;
    outfile.open(predict_data_save_path);
    for (int i = 0; i < (int) out_len; i++) {
        outfile << predict_result[i] << endl;
    }
    outfile.close();

    // free memory
    XGDMatrixFree(test_data);
    XGBoosterFree(booster);
    return 0;
}