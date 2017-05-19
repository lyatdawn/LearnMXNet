/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm-inl.h
 * \brief
 * \author
*/
#ifndef MXNET_OPERATOR_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_BATCH_NORM_INL_H_

#include <dmlc/logging.h> // mxnet的日志头文件. 在dmlc-core/include/dmlc下, 
#include <dmlc/parameter.h> // mxnet的参数头文件, 在dmlc-core/include/dmlc下, 定义参数的. 
#include <mxnet/operator.h> // 在include/mxnet下, 定义操作基类(operator), 操作属性类, 方法等. 对OP或Prop的函数进行声明. 
#include <map> // 关联式容器, 元素的值与某个特定的键相关联, 而并非通过元素在数组中的位置类获取. 
#include <vector> // 向量容器. 
#include <string> // 字符串. 
#include <utility> // utility头文件定义重载的关系运算符, 简化关系运算符的写入, 还定义了pair类型,
// pair类型是一种模板类型, 可以存储一对值. 
#include "./operator_common.h" // src/operator下, mxnet的层一些常用的属性.
#include "./mshadow_op.h" // src/operator下, 定义了一些结构体. 这些结构体用来接收数据实现某些层的前向输出和反向输出, 如激活函数 
// 层有softplus, softplus_grad. 一个计算前向的输出, 一个计算反向的输出. 

#include <iostream>

using namespace std;

namespace mxnet {
namespace op {

namespace batchnorm {
enum BatchNormOpInputs {kData, kGamma, kBeta}; // BN层输入参数, kData为0, kGamma为1, kBeta为2. 这里批训练时, gamma和beta的值可
// 以对所有batch的样本一样, 也可以不一样,  
enum BatchNormOpOutputs {kOut, kMean, kVar}; // BN层的输出参数, kOut为0, kMean为1, kVar为2. 利用kData可以首先计算出kMean和kVar
// 然后在此基础上, 联合kGamma和kBeta计算kOut. (用符号代替了变量). 
enum BatchNormOpAuxiliary {kMovingMean, kMovingVar}; // BN操作的辅助变量, kMovingMean为0, kMovingVar为1. 在做前向操作时能更好
// 地理解这两个量. 为求解batch数据的Mean和Var服务. 为了方便计算而需要的附加的tensor. 
enum BatchNormBackResource {kTempSpace}; // 反向传播的资源配置, 设置一个临时空间, 这个空间可以是任意大小的. 
/*
有些操作需要额外的内存作为工作空间进行计算, 比如说BatchNormBackward. 这种情况下, 
系统最好可以对这部分内存进行管理, 这样系统可以做一些优化, 比如说内存的重复利用.
struct ResourceRequest {
  enum Type {
    kRandom,  // get an mshadow::Random<xpu> object
    kTempSpace,  // request temporay space
  };
  Type type;
};
*/ 
}  // namespace batchnorm

struct BatchNormParam : public dmlc::Parameter<BatchNormParam> { // BatchNormParam, BN操作参数结构体, 对BN层的参数进行描述, 设
// 置初值, 设定范围等. 
  float eps; // eps, 即BN操作中从 x^(k) --> X^(k)时, 要x^(k)减去批样本均值, 除以批样本方差, 除以方差时为防为0, 变为
  // var[x^(k)] + eps.  
  float momentum; // momentum, momentum是moving average的动量项. float, 初值是0.9f.  
  bool fix_gamma; // fix_gamma, bool. 在训练过程中是否固定伸缩因子gamma. 
  bool use_global_stats; // bool.  
  bool output_mean_var; // bool. 是否输出样本均值和方差.  
  DMLC_DECLARE_PARAMETER(BatchNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0"); // epsilon, DMLC_DECLARE_FIELD宏, 输入参数是eps. set_default设置初值为1e-3f, 
	// describe描述函数.  
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average"); // momentum初值为0.9f.  
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Fix gamma while training"); // 在训练网络时, 默认固定缩放因子gamma.  
    DMLC_DECLARE_FIELD(use_global_stats).set_default(false)
    .describe("Whether use global moving statistics instead of local batch-norm. "
              "This will force change batch-norm into a scale shift operator.");
    /*
	对于use_global_stats, 参考(caffe+报错︱深度学习参数调优杂记+caffe训练时的问题+dropout/batch Normalization ).
	use_global_stats == true时会强制使用模型中存储的BatchNorm层均值与方差参数, 而非基于当前batch内计算均值和方差. 
	
	而BatchNormlization文章介绍的BN方法, 训练过程中是基于mini-batch的. , BN是基于mini-batch的:
	对于一个mini-batch的输入{x1, x2, ..., xm}, 通过这m个输入来计算mean, var. 然后计算 xi^(~), 即相当于是BN层真正的输入. 在计算
	BN层的输出y^(i). 输入{x1, x2, ..., xm}是一个batch的输入, 而不是整个数据集的.
	use_global_stats == true时, 就相当于是使用整个数据集的计算结果{x1, x2, ..., xN}做为BN前一层的输入.
	而在测试阶段, 均值和方差已经不是针对某一个Batch了, 而是针对整个数据集而言. 因此, 在训练过程中除了正常的前向传播和反向求导
	之外, 我们还要记录每一个Batch的均值和方差, 以便训练完成之后按计算整体的均值和方差.  
	
	网络前向传播, 一次性输入一个batch的数据; 然后再反向传播. 对于一个batch的数据, 网络迭代T次终止, 再进行写一个bath数据的迭代.
	即, 对于每一个batch的数据, 网络迭代T次. 对于整个数据集, 网络一共迭代epoch次.     
	*/
    
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
    .describe("Output All,normal mean and var"); // 默认不输出数据的均值和方差.  
  }
};

/*
一般BN layer放在FC和conv的后面, 因此dlib做了bn_fc和bn_con层.  
*/

template<typename xpu> // 模板参数只有xpu.  
class BatchNormOp : public Operator { // BatchNormOp, BN操作类.   
 public:
  explicit BatchNormOp(BatchNormParam param) {
    /*
	BatchNormOp, BN操作类的构造函数: C++中的explicit关键字只能用于修饰只有一个参数的类构造函数, 它的作用是表明该构造函数是显示
	的, 而非隐式的. param是BN参数类的对象, 利用param来访问BN的参数.  
	*/
    this->param_ = param; // BatchNormParam param_, 生成BatchNormParam结构体的一个副本.  
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    /*前向操作, 虚函数. 函数的实现在类中定义. 不需要返回值. 本层为第 l 层. 
	in_data: 本层输入data, 包括kData, kGamma, kBeta.
	req: 数据操作模式. 
	out_data: 本层输出, out. 在训练的时候本层输出有两个.  
	aux_states: 表示的是为了方便计算而需要的附加的 tensor. 附加的Tensor有两个: kMovingMean, kMovingVar. 以前看的操作均没使用
	aux_states来辅助计算. 
	*/
    using namespace mshadow;
    using namespace mshadow::expr;
    
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(aux_states.size(), 2);
    /*
	in_data容器大小是3, 即有三个Tensor, 包括kData, kGamma, kBeta.
	aux_states容器大小是2, 即有两个附加的Tensor, 包括kMovingMean, kMovingVar.    
	*/
    
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 3);
      CHECK_EQ(req.size(), 3);
      /*
	  ctx是OpContext结构体定义的成员. OpContext结构体定义见include/mxnet/operator.h. 利用ctx成员访问结构变量is_train:
	  int is_train; // operator是在进行 train 还是 test (is_train); 
	  
	  在训练阶段, out_data的容器大小是3, 即根据BN层的输入, 要计算mean, var, out. 想用的数据操作模式也是3个.  
	  */
      
    } else {
      CHECK_GE(out_data.size(), 1);
      CHECK_GE(req.size(), 1);
      CHECK_EQ(req[batchnorm::kOut], kWriteTo);
      /*
	  在网络的test/predict阶段, out_data容器大小为1. BN层的输出只有输出out. 相应的数据操作模式也是1个. 而且数据操作模式是:
	  kWriteTo, 即out代表的 tensor 提供的是可以直接写入的原始的内存块. 
	  */
    }

    Stream<xpu> *s = ctx.get_stream<xpu>();
    const real_t scale = static_cast<real_t>(in_data[batchnorm::kData].shape_[1]) /
                         static_cast<real_t>(in_data[batchnorm::kData].shape_.Size());
    /*
	static_cast < type-id > ( expression ), C++新标准定义的四个转换符, 即static_cast, dynamic_cast, reinterpret_cast和
	const_cast. static_cast该运算符把expression转换为type-id类型, 但没有运行时类型检查来保证转换的安全性. 即将expression转换成
	real_t型的, 即float型. 

	in_data[batchnorm::kData].shape_[0]: 65 第一维是batch_size的大小. 
	in_data[batchnorm::kData].shape_[1]: 10 第二维是BN前一层特征图的个数.  
	in_data[batchnorm::kData].shape_[2]: 47 
	in_data[batchnorm::kData].shape_[3]: .. 第三维和第四维是数据的大小. 
	in_data[batchnorm::kData].shape_.Size():  427700
	
	如果BN层前一层是FC层, shape_[0]为batch_size; shape_[1]为FC层的结点个数, 可以这样理解, 一个结点就是一个特征图. 
	
	shape_.Size()就是in_data[batchnorm::kData]即BN层输入数据各个维度的乘积. 即输入数据的总个数. 
	
	scale是real_t类型的(float类型), 其值等于: 前一层特征图(结点)的个数 / 一个batch输入数据(BN层的输入数据)的总个数.   
	*/
	/*cout<<"in_data[batchnorm::kData].shape_[0]: "<<in_data[batchnorm::kData].shape_[0]<<endl;
	cout<<"in_data[batchnorm::kData].shape_[1]: "<<in_data[batchnorm::kData].shape_[1]<<endl;
	cout<<"in_data[batchnorm::kData].shape_[2]: "<<in_data[batchnorm::kData].shape_[2]<<endl;
	cout<<"in_data[batchnorm::kData].shape_[3]: "<<in_data[batchnorm::kData].shape_[3]<<endl;
	cout<<"in_data[batchnorm::kData].shape_.Size(): "<<in_data[batchnorm::kData].shape_.Size()<<endl;*/
	
    Tensor<xpu, 4> data; // data, xpu下的4维张量. 
    Tensor<xpu, 4> out; // out, xpu下的四维张量. 
    if (in_data[batchnorm::kData].ndim() == 2) { // 如果in_data[batchnorm::kData]即BN层的输入数据是2维的, 那么需要先定义dshape
	  // 然后再将in_data[batchnorm::kData]拉成4维的张量.
	  /*====================================================================================================================== 
	  BN层前为FC层, 设FC层结点个数是 num_hidden, 那mean和var的维数为num_hidden, 然后将mean扩充成 batch_size * num_hidden *
	  1 * 1的再执行 data - mean的操作. 即mxnet写的batch_norm-inl.h的代码对于前一层是FC层同样适用, 可以将FC的输出out扩展成
	  batch_size * num_hidden * 1 * 1的, 再作为BN层的输入. BN层的前一层是FC层时, 理论和实际是可以结合起来的.
	  
	  如BN层前为FC层, 那么in_data[batchnorm::kData].ndim() == 2, 要想对FC的激活值使用BN操作, 就要先将FC的激活值data拉成4维的
	  张量, 大小为: batch_size * num_hidden *1 * 1.  
	  */ 
      Shape<4> dshape = Shape4(in_data[batchnorm::kData].shape_[0],
                               in_data[batchnorm::kData].shape_[1], 1, 1);
      /*
	  定义dshape, 4维shape. Shape4定义:
	  MSHADOW_XINLINE Shape<4> Shape4(index_t s0, index_t s1, index_t s2, index_t s3){
	      Shape<4> s;
          s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
  		  return s; } 
  	  s0 = in_data[batchnorm::kData].shape_[0], 即batch_size, dshape[0]; s1 = in_data[batchnorm::kData].shape_[1], 即BN层前一层
	  特征图的个数, 如果是前连接层这种的, 就是结点个数, dshape[1]; s3 = s4 =1, dshape[2], dshape[3].  	  
	  */
	  
      data = in_data[batchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[batchnorm::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      /*
	  将in_data[0](输入数据)拉成4维的张量. 这里将TBlob数据拉成Tensor数据时没有使用FlatTo2D, 而是用了get_with_shape. 定义如下:
	  mshadow::Tensor<Device, dim, DType> mxnet::TBlob::get_with_shape(const mshadow::Shape<dim> & shape, 
	  mshadow::Stream< Device > *stream = NULL)const. 给定shape, 将TBlob拉成一个Tensor. 如果shape和存储的大小不一致时, 会报错.
	  
	  定义BN层的输出out, 将out_data[batchnorm::kOut]用了get_with_shape拉成4维张量. 
	  */
      
    } else {
      data = in_data[batchnorm::kData].get<xpu, 4, real_t>(s);
      out = out_data[batchnorm::kOut].get<xpu, 4, real_t>(s);
      /*
	  BN层的输入数据不是2维的, 就是4维的. 就直接使用get函数将in_data[batchnorm::kData], out_data[batchnorm::kOut]拉成4维的张量.
	  mshadow::Tensor<Device, dim, DType> mxnet::TBlob::get(mshadow::Stream<Device> *stream = NULL)const. 
	  */
    }// if else语句执行的结果是类似的, 均是定义4维张量data和out. 区别是BN层的前一层, 根据输入数据的维数来确定data和out如何确定. 
    
    Tensor<xpu, 1> slope = in_data[batchnorm::kGamma].get<xpu, 1, real_t>(s); // gamma. 
    Tensor<xpu, 1> bias = in_data[batchnorm::kBeta].get<xpu, 1, real_t>(s); // beta. 
    Tensor<xpu, 1> moving_mean = aux_states[batchnorm::kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[batchnorm::kMovingVar].get<xpu, 1, real_t>(s);
    /*
	利用get函数将:
	in_data[batchnorm::kGamma]即in_data[1]拉成1维的张量, 即向量. slope. 是原文中的gamma.
	in_data[batchnorm::kBeta]即in_data[2]拉成1维的张量. bias. 是算法中的beta.
	aux_states[batchnorm::kMovingMean]即aux_states[0]拉成1维的张量, moving_mean. 
	aux_states[batchnorm::kMovingVar]即aux_states[1]拉成1维的张量, moving_var. 
	
	aux_states容器中的数据是做辅助计算的. 获取moving average, 如果use_global_stats == true, 那么就要使用 moving average.  
	moving_mean和moving_var有初值, 在反向传播过程中会根据BN层的输出均值mean和方差var进行更新. 
	*/
	
    if (param_.fix_gamma) slope = 1.f; // 如果再训练阶段固定gamma, 那么就直接令slope = 1.f. 
    
    /*
	求BN层输入的均值mean和方差var是基于mini-batch的, 即让一个batch的输入数据{x1, x2, ..., xm}具有0均值, 1方差. 不针对单个样本! 
	即不是对一个样本的输入xi, 进行求均值mean和方差var: mean = 1/n * (sum( xij )), 再计算yi. 
	mean = 1 / m * (sum( xi )), xi是一个batch中BN层的输入. 
	
	BN层的输入数据是xi, 输出数据是yi, 中间变量时xi^(^). 
	*/

	/*====================================================================================================================  
    一般BN layer放在FC和conv的后面, 因此dlib做了bn_fc和bn_con层. BN层的输入data是4维的张量, 输出也是4维的张量. 
	1>BN层前为conv层, 设卷积层特征图的个数是n个, 那么mean和var是n维的向量, 与特征图的个数是相对应的; 然后再计算 xi^(^)时, 
	先将mean扩充成 batch_size * n * Nh * Nw(Nh是卷积层特征图的高度, Nw是卷积层特征图的宽度); 然后才可以进行 data - mean的
	操作. 但是从 batch * n个特征图得到mean]和var的过程没太想明白.
	  
    2>BN层前为FC层, 设FC层结点个数是 num_hidden, 那mean和var的维数为num_hidden, 然后将mean扩充成 batch_size * num_hidden *
	1 * 1的再执行 data - mean的操作. 即mxnet写的batch_norm-inl.h的代码对于前一层是FC层同样适用, 可以将FC的输出out扩展成
	batch_size * num_hidden * 1 * 1的, 再作为BN层的输入. BN层的前一层是FC层时, 理论和实际是可以结合起来的. 
	
	对于前一层是FC层, BN层的输入数据是2维的: batch_size * num_hidden. 因此要将输入data先拉成batch_size * num_hidden * 1 * 1的.
	输入data和输出out的大小是一致的; 
	对于前一层是卷积层, 则对data直接使用BN操作即可.
	
	前向传播是这样, 反向传播也是这样. 
	======================================================================================================================*/
	
    // whether use global statistics
    if (ctx.is_train && !param_.use_global_stats) { // 网络在训练阶段. 而且不使用use global statistics. 即在训练阶段不使用
	  // use_global_stats, 否则网络不能收敛. 训练阶段基于mini-batch做BN处理, 针对当前 mini-batch 计算期望和方差. 
      Tensor<xpu, 1> mean = out_data[batchnorm::kMean].get<xpu, 1, real_t>(s);
      Tensor<xpu, 1> var = out_data[batchnorm::kVar].get<xpu, 1, real_t>(s);
      /*利用get函数将:
	  out_data[batchnorm::kMean]即out_data[1]拉成1维的张量. 保存BN输入的计算均值(激活值的均值). 
	  out_data[batchnorm::kVar]即out_data[2]拉成1维的张量. 保存BN输入的计算方差(激活值的方差). 
	  */
	  
	  /*==================================================================================================================== 
	  1)mean和var均是1维的张量, 即向量. 虽然是向量, 但是可以当做标量来用, 即 mean = 1.f是正确的. 
	  ======================================================================================================================*/
      
      CHECK(req[batchnorm::kMean] == kNullOp || req[batchnorm::kMean] == kWriteTo);
      CHECK(req[batchnorm::kVar] == kNullOp || req[batchnorm::kVar] == kWriteTo);
      /*
	  BN输入的计算均值和方差的数据操作模式是kNullOp或者kWriteTo(tensor可以直接写入的原始的内存块).  
	  */
      
      // The first three steps must be enforced.
      mean = scale * sumall_except_dim<1>(data);
      var = scale * sumall_except_dim<1>(F<mshadow_op::square>(
          data - broadcast<1>(mean, data.shape_)));
      /*
	  在网络的训练阶段, 首先基于mini-batch计算BN输入的均值mean和方差var. scale是real_t类型的
	  (float类型), 其值等于: 前一层特征图(结点)的个数 / 一个batch输入数据(BN层的输入数据)的总个数. 例如对于BN层前一层是FC层, 
	  scale = 1 / batch_size; 对于卷积层scale = 1 / (batch_size * 输入数据维数乘积). 
	  
	  data是BN层的输入数据, 包含了一个batch的数据. data是4维的张量, data[0]是batch_size, 即样本个数; data[1]是一个样本的
	  channel, 是1还是3(3维的不能计算); data[2]是一个样本的高度(矩阵的行数); data[3]是一个样本的宽度(矩阵的列数). 
	  1)mean:
	  mean = scale * sumall_except_dim<1>(data); scale是一个数, 扮演原文Algorithm1中的 1 / m. 
	  
	  sumall_except_dim定义见mshadow/mshadow/extension/reduceto1d.h44行:
	  template<int dimkeep,  typename SrcExp, typename DType, int etype>
	  inline ReduceTo1DExp<SrcExp, DType, red::sum, ExpInfo<SrcExp>::kDim - dimkeep> sumall_except_dim(const Exp<SrcExp, 
	  DType, etype> &exp){...}. sumall_except_dim的功能是对除dimkeep维度外, 所有exp的维度进行求和. 
	  返回expresion with type Tensor<Device,1>. 参数:
	  exp: 输入表达式, 必须是一个Tensor<?,2>, 即一个矩阵.
	  dimkeep: 需要保留的exp维度. 维度从0开始计算. 
	  
	  sumall_except_dim<1>(data)即对一个batch的所有数据求和(不管data[1]), 是数据矩阵的和. 即sum( xi ), 
	  xi对应BN层的输入数据矩阵. sum( xi )即矩阵的加法. 
	  
	  这句代码执行的就是: mean = (1 / m) * sum( xi ). 
	  
	  2)var:
	  该句代码执行的就是:
	  var = (1 / m) * sum( xi - mean). 
	  scale是一个数, 扮演原文Algorithm1中的 1 / m.
	  sum( xi - mean)为: sumall_except_dim<1>(F<mshadow_op::square>(data - broadcast<1>(mean, data.shape_))).
	  
	  F<mshadow_op::square>(a)是一个单目运算符, 运算符是mshadow_op::square, 见src/operator/mshadow_op.h下的struct square, 
	  输入DType a, return DType(a * a). 其中a是: data - broadcast<1>(mean, data.shape_), 即 xi - mean, BN层的每个输入 - batch
	  个样本输入的均值. 
	  
	  broadcast<1>(mean, data.shape_), broadcast见: mshadow/mshadow/extension/broadcast.h 69行:
	  template<int dimcast, typename SrcExp, typename DType, int etype, int dimdst>
	  inline Broadcast1DExp<SrcExp, DType, dimdst, dimdst - dimcast> broadcast(const expr::Exp<SrcExp, DType, etype> &src, 
	  Shape<dimdst> shape) {..}. 
	  src Tensor<Device,1>; shape: shape of output; 返回 a expresion with type Tensor<Device, dimdst>, dimdst为4, 
	  返回的Tensor的维数为4, 和shape的个数是有关的.
	  * input: Tensor<Device,1>: ishape[0]
	  * output: Tensor<Device,dimdst> : oshape[dimcast] = ishape[0].
	  将一个1维的 Tensor 扩充成 dimdst 维的 Tensor. 为了正确计算!! 
	  
	  mean是几维的Tensor才能正确说明问题!! 
	  
	  为了计算xi - mean, BN层的每个样本的激活值 - batch个激活值的均值. 进行的操作是: data - broadcast<1>(mean, data.shape_), 
	  因此需要将mean扩充到和data一样大小才能进行正确地减法. broadcast<1>(mean, data.shape_)就是将mean(1维的Tensor)扩充成和
	  data一样大小的Tensor, 即Tensor<xpu, 4>.  即 (broadcast<1>(mean, data.shape_))[0]为Batch_size; 
	  (broadcast<1>(mean, data.shape_))[1]为channel; (broadcast<1>(mean, data.shape_))[2]为data的高度;
	  (broadcast<1>(mean, data.shape_))[3]为data的宽度. 因此, data - broadcast<1>(mean, data.shape_)就是
	  BN层的每个样本的激活值 - batch个激活值的均值, 即x1 - mean, x2 - mean, ..., xm - mean.
	  
	  然后对data - broadcast<1>(mean, data.shape_)平方做和再取scale即可. 求和时和求mean时的做法一致, 可以看做是 
	  (x1 + mean) + (x2 + mean) + ... + (xm - mean). 
	  */    
          
      Assign(out, req[batchnorm::kOut], broadcast<1>(slope, out.shape_) *
             (data - broadcast<1>(mean, data.shape_)) /
             F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_)) +
             broadcast<1>(bias, out.shape_));
      /*
	  Assign赋值操作, out是BN层的输出, req是数据操作模式, exp即 gamma * [(data - mean) / (var + eps)^(1/2)] + beta, 
	  gamma即slope, beta即bias. 
	  
	  exp为: broadcast<1>(slope, out.shape_) * (data - broadcast<1>(mean, data.shape_)) 
	  / F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_)) 
	  + broadcast<1>(bias, out.shape_) 
      首先将slope扩充成和out具有相同shape的Tensor(gamma); 再乘(data - broadcast<1>(mean, data.shape_)), 即data - mean;
	  然后F<mshadow_op::square_root>是一个单目运算符, 运算符是mshadow_op::square_root, 结构体mshadow_op::square_root输入
	  (DType a, 返回DType(sqrtf(a)), 即float型的a^(1/2), 即对broadcast<1>(var + param_.eps, data.shape_)做开放操作, 
	  broadcast<1>(var + param_.eps, data.shape_)即(var + eps), 这里, var是1维的张量, 可以当做标量用, 因此var + eps有效. 
	  (var + eps)的结果还是Tensor<xpu, 1>的, 因此再将(var + eps)扩充成和data具有一样shape的Tensor.; 最后加上beta,
	  即broadcast<1>(bias, out.shape_), 将bias扩充成和out具有一样shape的Tensor.
	  */
      
    } else {
      /*
	  在train阶段, 对每一个minibatch使用BN, 那么, 在test/predict的时候怎, 常见的做法是使用整个train-set计算出mean. 
	  由于train-set的数据量非常大, 计算mean计算量非常大, 所以经常采用的技术是使用moving average算法, 在为此在训练过程中需要记录
	  每一个Batch的均值和方差, 以便训练完成之后按照下式计算整体的均值和方差:
	  E[x] = Eb[meanb]; Var[x] = (m / (m - 1)) * Eb[varb].
	  meanb是第b个batch的mean, varb是第b个batch的var.
	  
	  在test/predict阶段, 或者是use_global_stats == true时(这两者其实可以看成是一种情况, 在训练阶段, use_global_stats == false
	  否则网络是不收敛的). 使用moving average算法来估计整个测试集的mean和var. 
	  
	  在统计学中, moving average算法是通过创建数据集的一系列不同子集的均值来分析数据的. 
      MovingAverage可翻译为滑动平均或移动平均, 是做时间序列预测时用到的简单方法. 
	  计算方法: 对于一个给定的数列, 首先设定一个固定的值k, 然后分别计算第1项到第k项, 第2项到第k+1项, 第3项到第k+2项的平均值, 
	  依次类推. 
	  */ 
      Assign(out, req[batchnorm::kOut], broadcast<1>(slope /
                                          F<mshadow_op::square_root>(moving_var + param_.eps),
                                          data.shape_) * data +
             broadcast<1>(bias - (slope * moving_mean) /
                          F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
      /*
	  Assign赋值操作, out是BN层的输出, req是数据操作模式, exp即 gamma / (var[x] + eps)^(1/2) * x + 
	  (beta - gamma * E[x] / (var[x] + eps)^(1/2)). 代码实现时, x即data, 为了使得能和data进行计算, 要对一些式子进行扩展, 扩展成
	  和data具有同样大小的shape. 由于使用了moving average算法, 因此用 moving_var 替代var, 用moving_mean替代mean. 将式子写为:
	  *1 + *2. 
	  slope即gamma, bias即beta. 
	  
	  令 a = F<mshadow_op::square_root>(moving_var + param_.eps), F<mshadow_op::square_root>即单目开方运算, moving_var是1维张量,
	  和eps相加. 然后利用broadcast<1>(), 将 slope / a 扩展成和data具有同样shape的Tensor, 即 
	  broadcast<1>(slope / a, data.shape_), 然后再和 data 相乘, 即可得 *1.
	  
	  令 b = F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_), 这和a是一样的. 然后执行 
	  bias - (slope * moving_mean) / b, 再将结果用broadcast<1>()展成和data具有同样shape的Tensor, 即 *2.    
	  */
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    /*BN层(第l层)有参数gamma和beta, 因此要计算的是损失J关在BN层(第l层)的残差, gamma的梯度和beta的梯度. 
    !!!!!!!!!!!!!!!!梯度可以看做是损失J关于层参数的导数, 残差可以看做是损失J关于层输入的导数!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
	 
    in_grad输出残差/梯度参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的.
	out_grad输入残差/梯度参数, 向量容器, 每个元素的类型是TBlob. 上一层(第l + 1层)的残差/梯度, 计算本层的残差/梯度. 
	in_data输入参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的输入.  
	out_data输出参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的输出. 	
	req: 数据操作模式, 向量数组. 元素类型是OpReqType.
	aux_states: 表示的是为了方便计算而需要的附加的 tensor. 附加的Tensor有两个: kMovingMean, kMovingVar. 以前看的操作均没使用
	aux_states来辅助计算.
	*/
	
	/*==================================================================================================================== 
	对BN层的求导可以发现, 有很多中间变量会重复使用. 这些中间变量可以单独算出来. 不过这也涉及到一个计算速度和存储之间的平衡
	问题. 
	*/   
							  
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), param_.output_mean_var ? 3 : 1);
    // bool output_mean_var; 是否输出样本均值和方差. 上一层的输入残差, 如果output_mean_var == true, out_grad有三个变量: 梯度,
	// mean, var; 否则, out_grad只有残差这一个.  
    CHECK_EQ(in_data.size(), 3); // BN层输入有三项, data输入, gamma, beta. 
    CHECK_EQ(out_data.size(), 3); // BN层输入有三项: out输出, mean均值, var方差. 
    CHECK_EQ(in_grad.size(), 3); // BN层的残差有三项, gslope即gamma的残差, gbias即beta的残差, grad_in即损失关于BN层的残差. 
    // grad_in, 损失J关于BN层输出的残差, 这个残差并不会对下一次的FC层的前向传播产生影响, 但是会利用gdata计算BN 
	// 层前一层(第l - 1)层的残差. 
    
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data, grad, grad_in; // 定义data, grad, grad_in. xpu下的4维张量. 下面会对这三个变量进行赋值. 
    const real_t scale = static_cast<real_t>(out_grad[batchnorm::kOut].shape_[1]) /
                         static_cast<real_t>(out_grad[batchnorm::kOut].shape_.Size()); // real_t scale, 与Foeward的一样.
	// 一层特征图(结点)的个数 / 一个batch输入数据(BN层的输入数据)的总个数. 例如对于BN层前一层是FC层, 
    // scale = 1 / batch_size; 对于卷积层scale = 1 / (batch_size * 输入数据维数乘积).
	 
    if (in_data[batchnorm::kData].ndim() == 2) { // BN层的输入数据in_data[batchnorm::kData是2维的, 调用TBol下的ndim成员函数,
	// 返回TBlob对象的维数.
	/*
	如BN层前为FC层, 那么in_data[batchnorm::kData].ndim() == 2, 要想对FC的激活值使用BN操作, 就要先将FC的激活值data拉成4维的
	张量, 大小为: batch_size * num_hidden * 1 * 1. 反向传播时是一样的, 也要分输入数据是2维的还是4维的, 
	*/ 
      Shape<4> dshape = Shape4(out_grad[batchnorm::kOut].shape_[0],
                               out_grad[batchnorm::kOut].shape_[1], 1, 1); // 定义Shape<4>的dshape, 
	  // 大小为: batch_size * num_hidden * 1 * 1.  
      data = in_data[batchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad = out_grad[batchnorm::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad_in = in_grad[batchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      /*
	  对420行定义Tensor<xpu, 4> 对象data, grad, grad_in进行赋值和定义操作.
	  data: BN层的输入数据, 因为in_data[batchnorm::kData]是2维的数据, 因此调用TBlob的get_with_shape函数, 传入dshape即
	  大小为: batch_size * num_hidden * 1 * 1的shape, 将BN层输入扩展成4维的Tensor.
	  grad: BN上一层(第l + 1)层的残差, 因为in_data[batchnorm::kData]是2维的数据, 因此out_grad[batchnorm::kOut]也是二维的. 因此
	  先扩展成4维的Tensor.
	  grad_in: BN层的残差. in_grad[batchnorm::kData]也是2维的Tensor, 先扩展为4维的.  
	  */
      
    } else {
      data = in_data[batchnorm::kData].get<xpu, 4, real_t>(s);
      grad = out_grad[batchnorm::kOut].get<xpu, 4, real_t>(s);
      grad_in = in_grad[batchnorm::kData].get<xpu, 4, real_t>(s);
      /*
	  如果in_data[batchnorm::kData].ndim()不是2维的数据, 那么就是4维的. 利用get函数直接将in_data[batchnorm::kData]等拉成4维的
	  张量即可. 
	  */
    } // 这和前向传播的操作基本是类似的. 

    Tensor<xpu, 1> mean = out_data[batchnorm::kMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> var = out_data[batchnorm::kVar].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> slope = in_data[batchnorm::kGamma].get<xpu, 1, real_t>(s);
    /*
	利用get函数将:
	out_data[batchnorm::kMean]即out_data[1], BN层的输出均值mean拉成1维的Tensor, mean向量.
	out_data[batchnorm::kVar]即out_data[2], BN层的输出方差Var拉成1维的Tensor. var.
	in_data[batchnorm::kGamma]即in_data[1], BN层的gamma参数拉成1维的Tensor, slope. 
	*/
    
    // Tensor<xpu, 1> bias = in_data[kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gslope = in_grad[batchnorm::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gbias = in_grad[batchnorm::kBeta].get<xpu, 1, real_t>(s);
    /*
	BN层的残差有三项, gslope即gamma的残差, gbias即beta的梯度, grad_in即损失关于BN层的梯度.
	slope和bias在前向传播时, 是1维的Tensor, 因此在反向传播中, 其残差也是1维的张量.
	in_grad[batchnorm::kGamma]即in_grad[1], 损失J关于gamma的残差, 是1维的张量.
	in_grad[batchnorm::kBeta]即in_grad[2], 损失J关于BN层beta参数的残差, 是1维的张量. 
	*/
    
    // update moving avg
    Tensor<xpu, 1> moving_mean = aux_states[batchnorm::kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[batchnorm::kMovingVar].get<xpu, 1, real_t>(s);
    /*
	aux_states[batchnorm::kMovingMean]即aux_states[0]拉成1维的张量, moving_mean. 
	aux_states[batchnorm::kMovingVar]即aux_states[1]拉成1维的张量, moving_var. 
	
	aux_states容器中的数据是做辅助计算的. 获取moving average, 如果use_global_stats == true, 那么就要使用 moving average. 
	*/

    if (param_.fix_gamma) slope = 1.f; // 如果gamma是一个定值, 那么slope(gamma)就是1.f. 

    if (ctx.is_train && !param_.use_global_stats) { // 在网络的训练阶段且不使用use_global_stats. 
       // 在test/predict阶段, 或者是use_global_stats == true时(这两者其实可以看成是一种情况, 训练时, use_global_stats == false.
	  // 否则网络是不收敛的). 再使用moving average算法来估计整个测试集的mean和var.  
      
	  /*
	  get requested temp space. 获取所需的临时空间. 
	  有些操作需要额外的内存作为工作空间进行计算, 比如说BatchNormBackward. 这种情况下, 系统最好可以对这部分内存进行管理, 
	  这样系统可以做一些优化, 比如说内存的重复利用. 因此BN有kTempSpace. 即BN的反向操作会申请一个临时的资源空间, 这个空间任意. 
	  */
      Tensor<xpu, 2> workspace = ctx.requested[batchnorm::kTempSpace].get_space<xpu>(
          mshadow::Shape2(3, mean.shape_[0]), s);
      /*
	  OpContext: 结构体, 定义在include/mxnet/operator.h中, 该结构体可以记录操作在前向和后向传播中的信息. ctx是结构体OpContext定
	  义的对象, requested是OPContext结构体下的函数:
      // brief Resources requested by the operator
  	  std::vector<Resource> requested; // 用来返回操作所需的资源. 
      ctx.requested返回的是一个向量容器, ctx.requested[batchnorm::kTempSpace]即ctx.requested[0]返回一个Resource对象, 然后
	  Resource对象再调用get_space函数. 
	  
	  get_space函数定义见: include/mxnet/resource.h 90行: get_space函数是定义在Resource结构体下的函数: 
	  template<typename xpu, int ndim>
	  inline mshadow::Tensor<xpu, ndim, real_t> get_space(mshadow::Shape<ndim> shape, mshadow::Stream<xpu> *stream)const{...}
	  get_space用来获取Tensor所需的空间. 参数shape: 返回Tensor的Shape; stream: Device下的Tensor; 返回所需的Tensor.
	  
	  此处, shape是Shape2(3, mean.shape_[0]), 第一维是3, 第二维是mean.shape_[0], BN前一层为FC层时, 为num_hidden结点个数; 为
	  卷积层时, 为特征图的个数. stream是xpu下的对象s. shape是Shape2, 即Shape<2>, 因此ndim是2, 故返回所需的Tensor是2维的.
	  
	  workspace即为BN反向传播所需的2维的Tensor, 是一个临时空间, 额外内存.   
	  */    
          
      Tensor<xpu, 1> gmean = workspace[0];
      Tensor<xpu, 1> gvar = workspace[1];
      Tensor<xpu, 1> tmp = workspace[2];
      /*
	  1维的Tensor gmean, gvar, tmp. 用workspace, BN层反向传播的临时Tensor定义. 利用gmean, gvar, tmp是损失关于参数gamma, beta
	  的梯度, 然后可以用gmean, gvar来计算损失J关于BN层输入的残差.   
	  
	  输出workspace.shape_.Size()为3, workspace.shape_[0]为3, workspace.shape_[1]为1, workspace.shape_[2]为1.
	  3是在定义workspace时的Shape2的第一个参数. 即Tensor的第0个位置的元素均代表的是大小.  
	  */

      moving_mean = moving_mean * param_.momentum + mean * (1 - param_.momentum);
      moving_var = moving_var * param_.momentum + var * (1 - param_.momentum);
      /*
	  使用moving average算法, 更新mean和var, 用于test/predict. momentum是moving average的动量项, float, 初值是0.9f.
	  
	  moving_mean和moving_var有初值, 在反向传播过程中会根据BN层的输出均值mean和方差var进行更新. 在测试的前向传播过程中, 利用 
	  moving_mean和moving_var来代替整个测试集的均值和方差.
	  
	  更新规则即: a = a * momentum + a * (1 - momentum), momentum是moving average的动量项, float, 初值是0.9f.
	  moving_mean和moving_var均按照这个规则来更新. 
	  */
	  
	  /*
	  计算gmean和gvar, gmean和gvar用来计算损失J关于BN层输出的残差! gvar方差的梯度, gmean是均值的梯度.
	  根据原文, 设网络的损失为l, 那么需要计算一下偏导:
	  partial(l) / partial(xi^(^)) == partial(l) / partial(yi) * gamma.  
	  partial(l) / partial(varb), gvar. 
	  partial(l) / partial(meanb), gmean. 
	  partial(l) / partial(xi)
	  partial(l) / partial(gamma), gslope. 
	  partial(l) / partial(beta), gbias. 
	  varb即第b个batch个样本BN层的输出方差, meanb即第b个batch个样本BN层的输出均值. 
	  */
      gvar = sumall_except_dim<1>((grad * broadcast<1>(slope, data.shape_)) *
                                  (data - broadcast<1>(mean, data.shape_)) *
                                  -0.5f *
                                  F<mshadow_op::power>(broadcast<1>(var + param_.eps, data.shape_),
                                                       -1.5f));
      /*
	  计算损失关于方差var的梯度, 根据原文为: partial(l) / partial(varb) =  
      sum{ [partial(l) / partial(xi^(^))] * [(xi - meanb)] * -0.5 * (varb + eps)^(-3/2) }.
      sum{ [*1] * [*2] * -0.5 * (*3) }.   
      
	  而 [partial(l) / partial(xi^(^))] == partial(l) / partial(yi) * gamma. yi是BN层的输出, 即下一层的输入, 又残差是损失关于
	  输入的导数, 因此 partial(l) / partial(yi) 就是BN上一层(第l + 1)层的残差. 这个残差即grad, 是将out_grad[0]拉成4维Tensor.
	  因此, *1 就是grad * gamma. 因此要对slope(gamma)进行扩展, slope即BN层的gamma参数, 由于BN层的输入xi和yi的shape相同, 因此
	  grad和BN层输入data的shape相同, 对slope进行扩展, 即将slope这个1维的Tensor扩展成和BN层输入数据data具有一样shape的Tensor.
	  broadcast<1>(slope, data.shape_)是扩展后的slope. 最后 *1 = grad * broadcast<1>(slope, data.shape_). 
	  
	  *2 = (xi - meanb). 由于是批处理, xi即data, 因此为了正确执行(xi - meanb), 要对meanb进行扩展. mean是BN层的输出均值, 为1维
	  的Tensor, 因此需要将mean扩展成和data具有相同shape的Tesnor, 即4维的Tenso. *2 = data - broadcast<1>(mean, data.shape_).
	  
      *3 = (varb + eps)^(-3/2). 首先F<mshadow_op::power>(*11, *21)是双目运算符, 运算符是mshadow_op::power, 输入DType a, DType b
	  返回powf( a, b ). *21为-1.5f, 即float型的1.5.
	  *11是broadcast<1>(var + param_.eps, data.shape_), 即对var + param_.eps进行扩展, 扩展成和data具有相同shape的Tesnor, 
	  即4维的Tensor. var + param_.eps的运算结果还是1维的Tensor. 
	  
	  最后再对[*1] * [*2] * -0.5 * (*3)求和, 不管第一个维度, 对所有维度进行求和. 即对batch_size维度, 数据高度维度, 宽度维度
	  求和. 
	  */
	                                                   
      gmean = sumall_except_dim<1>(grad * broadcast<1>(slope, data.shape_));
      gmean *= -1.0f / F<mshadow_op::square_root>(var + param_.eps);
      tmp = scale * sumall_except_dim<1>(-2.0f * (data - broadcast<1>(mean, data.shape_)));
      tmp *= gvar;
      gmean += tmp;
      /*计算损失关于均值mean的偏导数, 根据原文为: partial(l) / partial(meanb) = 
	  sum{ [partial(l) / partial(xi^(^))] * [-1 / (varb + eps)^(1/2)] } 
	  + { [partial(l) / partial(varb)] * sum{ -2* [(xi - meanb)]} / m }. 即:
	  sum{ *1(求gvar时的*1) } * [*2] + { gvar * [*3] }. 由于gmean求时, 项比较多, 所以分开来求.
	  
	  首先令gmean = sum{ *1 }, *1为求gvar时的*1. 然后求和, 不管第一个维度, 对所有维度进行求和. 即对batch_size维度, 数据高度
	  维度, 宽度维度求和. 
	  
	  *2 = [-1 / (varb + eps)^(1/2)]. 这里计算varb + eps的结果是1维的Tensor. F<mshadow_op::square_root>()是单目开方运算. 1维的
	  Tensor可以看做是一个标量, 因此用-1f / F<mshadow_op::square_root>(). *2 还是一个1维的Tensor, 因此可以看做是一个标量, 
	  最后利用 gmean * (*2)即可!!
	  gmean = gmean * (*2)是 + 前的第一项.
	  
	  *3 = sum{ -2* [(xi - meanb)]} / m. 其中1/m用scale代替, 这和前向传播中的操作一样. 由于是批处理, 因此xi即data, 为了计算
	  data - mean, 要对mean即BN层的输出均值进行扩展, 扩展成和data具有相同shape的Tesnor, 即4维的Tensor, 这样就可以计算
	  data - mean. 再乘上 -2.0f, 然后和, 不管第一个维度, 对所有维度进行求和. 即对batch_size维度, 数据高度维度, 宽度维度求和.
	  *3 即tmp, tmp是1维的Tensor.  即 sumall_except_dim<1> 的计算结果是返回1维的Tenso.	     
	  
	  *3 * gvar即是 + 后面的那一项, 即tmp. 
	  因此, 损失关于BN层输出均值mean的梯度就是 gmean = gmean + tmp. 
	  */
	  
      // assign
      if (!param_.fix_gamma) { // 如果没有固定gamma值, 来计算损失J关于参数gamma的梯度. 
        Assign(gslope, req[batchnorm::kGamma],
               sumall_except_dim<1>(
                   grad * (data - broadcast<1>(mean, data.shape_)) /
                   F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_))));
        /*
		Assign赋值操作, gslope是损失关于BN层参数gamma的梯度; req是数据操作模式, 是kGamma的数据操作模式; exp, 根据原文:
		partial(l) / partial(gamma) = sum{ [partial(l) / partial(yi)] * xi^(^) }. 即:
		sum{ grad * xi^(^) }, xi^(^)是中间计算结果, 需要根据xi, mean, var算出来!!
		
		xi^(^) = [xi - meanb] / [varb + eps]^(1/2). 由于是批处理操作, xi即data. 因此想进行[xi - meanb], 需要扩展BN层的输出均值
		mean, 扩展成和data具有相同shape的Tesnor, 即4维的Tensor, 这样就可以计算data - mean.
		F<mshadow_op::square_root>(*)是单目开方运算. *是[varb + eps], 即先对BN层的输出方差var和eps求和, 然后再对这个1维Tensor
		进行扩展, 扩展成和data具有相同shape的Tesnor, 即4维的Tensor. 再做开方计算. 这样就可以得到xi^(^), 即data^(^).
		
		最后对 grad * data^(^)进行求和, 管第一个维度, 对所有维度进行求和. 即对batch_size维度, 数据高度维度, 宽度维度求和.   
		*/           
                   
      } else { // 固定了gamma, 即slope = 1.0f后, 损失关于gamma的梯度是0.0f. 因为gamma已经是定值了!! 
        Assign(gslope, req[batchnorm::kGamma], 0.0f);
      }
      Assign(grad_in, req[batchnorm::kData],
             (grad * broadcast<1>(slope, data.shape_)) *
             broadcast<1>(1.0f / F<mshadow_op::square_root>(var + param_.eps), data.shape_) +
             broadcast<1>(gvar, data.shape_) * scale * 2.0f * (data - broadcast<1>(mean,
                                                                                   data.shape_)) +
             broadcast<1>(gmean, data.shape_) * scale);
      /*
      Assign赋值操作, grad_in是损失关于BN层输入的梯度; req是数据操作模式, 是kData的数据操作模式;
	  计算损失关于BN层的输入xi(data)的梯度, 根据原文:
	  partial(l) / partial(xi) == grad_in = [partial(l) / partial(xi^(^))] * [1 / (varb + eps)^(1/2)] 
	  + [partial(l) / partial(varb)] * 2 * [(xi - meanb)] / m
	  + [partial(l) / partial(meanb)] /m. 即:
	  
	  [grad * gamma] * [1 / (varb + eps)^(1/2)] + [partial(l) / partial(varb)] * 2 * [(xi - meanb)] / m
	  + [partial(l) / partial(meanb)] /m.
	  
	  [grad * gamma]上面已经求过了, 将slope扩展成和data具有一样shape的Tensor即可.
	  [1 / (varb + eps)^(1/2)]前面已经求过, 只是将-1转换为1即可. (varb + eps)结果是1维的Tensor, 可做标量用. 还有一点, 由于计算
	  grad_in时, 用到了grad, 其shape和data的shape一致. 因此, 所有的变量均需要扩展成和data具有一样shape的Tensor, 即4维的张量. 
	  [partial(l) / partial(varb)]即gvar, 再将gvar这个1维的张量扩展成和data具有一样shape的Tensor, 即4维的张量.
	  1/m用scale替换, (xi - meanb)上面也已经算过了. xi即data, 需要扩展meanb.
	  [partial(l) / partial(meanb)] 即gmean, 再将这个1维的张量扩展成和data具有一样shape的Tensor, 即4维的张量. 1/m用scale替换.  
	  */    
             
      Assign(gbias, req[batchnorm::kBeta], sumall_except_dim<1>(grad));
      /*
      Assign赋值操作, gbias是损失关于BN层参数beta的梯度; req是数据操作模式, 是kBeta的数据操作模式;
	  计算损失关于beta的梯度, 根据原文:
	  partial(l) / partial(beta) = sum{ partial(l) / partial(yi) }. partial(l) / partial(yi)即grad, 是损失关于BN层输出,
	  第l + 1层的输入的残差.  然后求和, 不管第一个维度, 对所有维度进行求和. 即对batch_size维度, 数据高度维度, 宽度维度求和. 
	  */
	  
    } else {
      // use global statistics with freeze moving mean and var. 在测试阶段或者使用use global statistics时的反向传播! 
      if (!param_.fix_gamma) { // 如果没有固定gamma值, 来计算损失J关于参数gamma的梯度.   
        Assign(gslope, req[batchnorm::kGamma],
               sumall_except_dim<1>(
                   grad * (data - broadcast<1>(moving_mean, data.shape_)) /
                   F<mshadow_op::square_root>(broadcast<1>(moving_var + param_.eps, data.shape_))));
        /*损失关于gamma的梯度. 
		Assign赋值操作, gslope是损失关于BN层参数gamma的梯度; req是数据操作模式, 是kGamma的数据操作模式; exp为:
		在测试阶段使用use global statistics时, 利用moving average算法, 即涉及到var用moving_var代替. 
		
		在测试阶段使用use global statistics时, 损失关于BN参数gamma的梯度和训练阶段时类似的, 只是var用moving_var代替.
		*/
		           
      } else { // 固定了gamma, 即slope = 1.0f后, 损失关于gamma的梯度是0.0f. 因为gamma已经是定值了!!
        Assign(gslope, req[batchnorm::kGamma], 0.0f);
      }
      Assign(gbias, req[batchnorm::kBeta], sumall_except_dim<1>(grad));
      /*损失关于beta的梯度. 
      Assign赋值操作, gbias是损失关于BN层参数beta的梯度; req是数据操作模式, 是kBeta的数据操作模式;
	  计算损失关于beta的梯度, 和训练阶段且不使用use global statistics的反向传播是一样的! 
	  */
	  
      Assign(grad_in, req[batchnorm::kData], (grad * broadcast<1>(slope, data.shape_)) *
             broadcast<1>(
                 1.0f / F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
      /*
      Assign赋值操作, grad_in是损失关于BN层输入的梯度; req是数据操作模式, 是kData的数据操作模式;
       
	  在测试阶段使用use global statistics时, 损失关于BN层输入的残差计算如下:
	  detla^(l + 1) = partial(l) / partial(xi^(^)) * [ 1 / (Var[x] + eps)^(1/2)].
	  
	  而partial(l) / partial(xi^(^)) = partial(l) / partial(yi) * gamma. 前面已经求过了.
	  Var[x]是对于整个测试集来说的方差, 这里用 moving_var 估计. 
	  
	  由于涉及到grad, 即损失关于第l + 1层输入的残差, 其shape和BN层输入data的shape一致. 因此要将所有的量扩展成和data具有相同
	  shape的Tensor, 即4维的张量. 
      */
    }
  }

 private:
  BatchNormParam param_;
};  // class BatchNormOp

template<typename xpu>
Operator *CreateOp(BatchNormParam param, int dtype);


#if DMLC_USE_CXX11
class BatchNormProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(Shape1(dshape[1]));
    out_shape->push_back(Shape1(dshape[1]));

    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[1]));
    aux_shape->push_back(Shape1(dshape[1]));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BatchNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "BatchNorm";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[batchnorm::kOut],
            out_data[batchnorm::kMean],
            out_data[batchnorm::kVar],
            in_data[batchnorm::kData],
            in_data[batchnorm::kGamma]
           };
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  int NumVisibleOutputs() const override {
    if (param_.output_mean_var) {
      return 3;
    }
    return 1;
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mean", "var"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"moving_mean", "moving_var"};
  }

  Operator* CreateOperator(Context ctx) const override {
      LOG(FATAL) << "Not Implemented.";
      return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
      std::vector<int> *in_type) const override;

 private:
  BatchNormParam param_;
};  // class BatchNormProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BATCH_NORM_INL_H_
