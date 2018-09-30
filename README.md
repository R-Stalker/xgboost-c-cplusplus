# xgboost-c-cplusplus

# 一、文件目录

xgboost——这里的xgboost文件是从xgboost源码中打包出来的供c/c++使用的文件夹。你也完全可以直接用整个xgboost源码包

dense.feature——用于预测的特征数据，维度是108维

model_file——已经训练好的xgboost模型,训练数据使用libsvm格式（我专门提训练数据格式是有原因的，这里有个大坑）

predict_xgb.cpp——加载训练好的模型，加载测试数据，产生预测结果



# 二、正确打开方式
Step1: compile the code. command: 

<pre><code>
g++ predict_xgb.cpp -I xgboost/include -I xgboost/rabit/include xgboost/lib/libxgboost.a xgboost/rabit/lib/librabit.a xgboost/dmlc-core/libdmlc.a -fopenmp -Wall
</code></pre>

Step2: run "./a.out"

# 三、凡事皆有因
python因为自身语言缺陷，在线上的效率太低，返回单条请求的时间最多控制在200ms左右，考虑效率和之后业务发展，决定将线上模块从python迁移到c/c++。

xgboost本身是c++实现的，提供了c_api.h接口供使用，难度不大，但是网上资料很少，也踩了不少坑。不过运气不错，前前后后用了两三天的时间搞定了。

