/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution-inl.h
 * \brief
 * \author
*/
#ifndef MXNET_OPERATOR_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_CONVOLUTION_INL_H_ // 宏 

#include <mxnet/io.h> // include/mxnet下, 声明了输入和输出数据结构和数据迭代器. 
#include <mxnet/base.h> // include/mxnet下, 声明mxnet的配置信息以及基本数据结构.
/*如是否支持OpenCV, CUDA, CUDNN, VS以及声明了mxnet命名空间下的变量的定义, 如cpu, gpu, index_t以及Save, Load等. 
*/ 
#include <mxnet/ndarray.h> // include/mxnet下, 
#include <mxnet/operator.h> // 在include/mxnet下, 定义操作基类(operator), 操作属性类, 方法等. 对OP或Prop的函数进行声明. 
#include <dmlc/logging.h> // mxnet的日志头文件. 在dmlc-core/include/dmlc下,
#include <dmlc/optional.h> // 在dmlc-core/include/dmlc下, 声明了class optional.  
#include <algorithm> // 标准C++算法库. 
#include <map> // C++ map容器头文件. 
#include <vector> // 向量容器. 
#include <string> // 字符串. 
#include <utility> // utility头文件定义重载的关系运算符, 简化关系运算符的写入, 还定义了pair类型,
// pair类型是一种模板类型, 可以存储一对值.
#include "./operator_common.h" // src/operator下, mxnet的层一些常用的属性.

#include<iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

namespace mxnet {
namespace op {

namespace conv {
enum ConvolutionOpInputs {kData, kWeight, kBias}; // 卷积层的输入索引, 包括前一层的输出数据data, 本层的权重wmat和偏置. 
enum ConvolutionOpOutputs {kOut}; // 卷积层的输出索引. 
enum ConvolutionOpResource {kTempSpace}; // 卷积层的资源配置, 设置一个临时空间, 这个空间可以是任意大小的. 
/*
有些操作需要额外的内存作为工作空间进行计算, 比如说cudnnConvolutionForward. 这种情况下, 
系统最好可以对这部分内存进行管理, 这样系统可以做一些优化, 比如说内存的重复利用.
struct ResourceRequest {
  enum Type {
    kRandom,  // get an mshadow::Random<xpu> object
    kTempSpace,  // request temporay space
  };
  Type type;
};
*/
enum ConvolutionOpCudnnTune {kOff, kLimited, kFastest}; // 利用GPU完成卷积操作时, 有三个索引. 没有使用, 因此convolution-inl.h
// 是cpu版的. 
}

/*
卷积层输出的特征图的大小, 输出特征图尺寸 = [(输入特征图尺寸 + 2 * pad - 卷积核尺寸)/步长](向下取整) + 1.

Matlab中的应用函数-conv2:二维卷积, 一维对应conv. 0.8版本的convolution-inl.h只能做2D卷积, 0.9版本的
convolution-inl.h可以做3D卷积, 而且将2D卷积核3D卷积结合在一起了.

卷积操作类型, 这和MATLAB的卷积函数conv和conv2是类似的, conv做向量间的卷积, 即一维卷积; conv2做二维卷积, 即矩阵卷积. 
w = conv(A, B,'shape')回卷积的一部分. 这一部分由shape参数指定:
full 返回全部卷积值(缺省). kFull,  为1.
same 返回卷积的中心部分, 与A有相同的大小. 
valid 仅返回卷积中的那些被计算而没有补零的部分, 不补零.. kValid, 为0. 

卷积的计算步骤:
(1)卷积核绕自己的核心元素顺时针旋转180度, 是顺时针旋转180, 不是做转置! 
(2)移动卷积核的中心元素，使它位于输入图像待处理像素的正上方. 根据shape的不同会选择不同的卷积方式.
(3)在旋转后的卷积核中，将输入图像的像素值作为权重相乘. 
(4)第三步各结果的和做为该输入像素对应的输出像素.

A=[1 2 3;4 5 6;7 8 9];
B=[1 2;3 4];

conv2(A, B, 'full'), 是做全卷积, 首先对B进行180度顺时针旋转, 然后对A进行补0操作. 补零的时候在A的外围进行补零. 
补零的行数 = 2 *(Nh - 1), 补零的列数 = 2 * (Nw - 1). 卷积核B大小为Nh * Nw. 
操作时的A为:   操作时B为: . 然后再对应元素相乘在相加即是卷积后的结果. 
0 0 0 0 0        4 3       
0 1 2 3 0        2 1 
0 4 5 6 0
0 7 8 9 0 
0 0 0 0 0

conv2(A, B, 'valid'), 返回卷积计算中没有补零部分的计算结果, 即是CNN中的卷积操作, 让卷积核对齐A即可. 不补零. 

conv2(A,B, 'same'), 返回和A同样大小的卷积后的结果, 利用full操作可以得到same操作的结果. 不在左上 
*/

struct ConvolutionParam : public dmlc::Parameter<ConvolutionParam> { // 卷积层的参数结构ConvolutionParam. 
  TShape kernel; // 卷积核, TShape: 一个shape类, 该类可以表示一个Tensor的形状. 利用TShape来表示Tensor的大小. 
  /*卷积核是二维的(h, w)或三维的(d, h, w). 即实现二维卷积或三维卷积, 对应到图像中可以是灰度图像(2维张量), RGB图像(3维张量).
  这和输入的数据是2维的还是3维的有关. (channel K不一定是图像的通道数了, 或者为前一层特征图的数量.) 
  TShape定义: http://mxnet.io/doxygen/classmxnet_1_1TShape.html*/  
  TShape stride; // 卷积核滑动步长, 二维的(h, w)或三维的(d, h, w).  
  TShape dilate;
  /*
  卷积核膨胀是将卷积核扩张到膨胀尺度约束的尺度中, 并将原卷积核没有占用的区域填充零. 例如:
  1 2 3      1 0 2 0 3
  4 5 6 ---->0 0 0 0 0      
  7 8 9      4 0 5 0 6
             0 0 0 0 0
             7 0 8 0 9
  卷积核由3*3膨胀到了5*5. 膨胀后的卷积核中填充了一些0. 膨胀系数与卷积核膨胀的关系, 首先回到卷积核膨胀公式:
  膨胀的卷积核尺寸 = 膨胀系数 * (原始卷积核尺寸 - 1) + 1.
  由于卷积的操作特性, 卷积核尺寸是奇数, 卷积核的膨胀系数刻画了卷积核高和宽方向的扩张倍数, 
  膨胀系数保证了膨胀的卷积核尺寸为奇数. 
  */
  TShape pad; // 原始数据的高度或宽度方向上补上0值的圈数, 二维的(h, w)或三维的(d, h, w).
  uint32_t num_filter; // 卷积核的个数. 经过卷积后一共有多少个特征图. 
  /*
  uint8_t，uint16_t，uint32_t: 使用typedef给类型起的别名.
  typedef unsigned char uint8_t; 1字节 
  typedef unsigned int uint16_t; 2字节 
  typedef unsigned long uint32_t; 4字节 
  typedef unsigned long long uint64_t; 8字节 
  */
  uint32_t num_group;
  /*
  将输入数据切割成num_group个partitions, 然后在每个partition上使用卷积操作, 再将卷积的结果连接起来. 有点并行的意思. 
  */
  uint64_t workspace; // 卷积操作最大的临时工作空间, 以MB计算. 
  bool no_bias; // 卷积层是否使用偏置, 默认使用偏置. 
  dmlc::optional<int> cudnn_tune;
  /*
  template<typename T>
  class dmlc::optional< T >
  C++17的可选类, 实例化对象cudnn_tune, cudnn_tune描述如下: 
  cudnn_tune决定是否通过测试性能来选择卷积算法, 由于要测试性能, 因此启动时间可能会变长, 但是操作的速度却会变快. 
  cudnn_tune有三个可选参数:
  off: 不调整
  limited_workspace: 在不超过工作区限制的情况下, 运行测试性能, 并选择最快的算法.
  fastest: 忽略工作区限制, 选择最快的算法. 
  */
  bool cudnn_off; // 该卷积层是否使用cudnn, 默认使用． 
  dmlc::optional<int> layout;
  /*
  layout与cudnn_tune类型一致. 设置输入, 输出和权重的layout(布局).  
  */ 
  DMLC_DECLARE_PARAMETER(ConvolutionParam) { // #define DMLC_DECLARE_PARAMETER(PType), 使用宏来描述卷积层的参数. 
    
	/*
	宏DMLC_DECLARE_PARAMETER, DMLC_DECLARE_FIELD的定义以及函数describe, set_default, set_range, add_enum均在
	#include <dmlc/parameter.h>下. 
	*/
	
    DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (h, w) or (d, h, w)");
    /*#define DMLC_DECLARE_FIELD(FieldName)使用宏#define DMLC_DECLARE_FIELD(FieldName)来对卷积层的参数进行描述, 利用函数
	describe(const std::string &description)对参数FieldName进行描述. 卷积核的大小, 二维卷积或三维卷积.*/
    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .describe("convolution stride: (h, w) or (d, h, w)");
    /*
	卷积核滑动步长, 二维滑动或三维滑动. 利用函数set_default(const DType &default_value)设置参数FieldName的默认值, 默认值是
	TShape(), 即mxnet::TShape::TShape( ), 调用TShape的默然构造函数, NONE; 利用函数describe(const std::string &description)
	对参数FieldName进行描述.  
	*/
	
	/*
	TShape a = TShape();
	cout<<"TShape()[0]: "<<a[0]<<endl; 0
	cout<<"TShape()[1]: "<<a[1]<<endl; 0
	cout<<"TShape()[2]: "<<a[2]<<endl; 0
	
	即TShape的默然构造函数TShape()构造的 shape对象, 其值均是0. 
	
	但是stride的默认值是(1, 1), 不可能是(0, 0). dliate和pad的默认值均值1.
	conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), stride=(2,2), num_filter=20), 指定stride=(2, 2).
	conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20), 不指定stride=(1, 1). 
	*/
	
    DMLC_DECLARE_FIELD(dilate).set_default(TShape())
    .describe("convolution dilate: (h, w) or (d, h, w)"); // 膨胀系数.  
    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("pad for convolution: (h, w) or (d, h, w)"); // 始数据的高度或宽度方向上补上0值的圈数.
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("convolution filter(channel) number");
    /*
	卷积核的个数. 调用函数set_range(DType begin, DType end)设置num_filter的范围, 1-100000. 
	*/
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions. Equivalent to slicing input into num_group\n    "
              "partitions, apply convolution on each, then concatenate the results"); // 将输入数据切割成num_group个partitions
    // 默然不切割输入. 
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
    .describe("Maximum tmp workspace allowed for convolution (MB).");
    /*卷积操作最大的临时工作空间, 默认是1024M, 范围是0M - 8192M.*/
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter."); // 卷积操作是否使用偏置. 
    DMLC_DECLARE_FIELD(cudnn_tune)
    .add_enum("off", conv::kOff)
    .add_enum("limited_workspace", conv::kLimited)
    .add_enum("fastest", conv::kFastest)
    .set_default(dmlc::optional<int>())
    .describe("Whether to pick convolution algo by running performance test.\n    "
              "Leads to higher startup time but may give faster speed. Options are:\n    "
              "\'off\': no tuning\n    "
              "\'limited_workspace\': run test and pick the fastest algorithm "
              "that doesn't exceed workspace limit.\n    "
              "\'fastest\': pick the fastest algorithm and ignore workspace limit.\n    "
              "If set to None (default), behavior is determined by environment\n    "
              "variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,\n    "
              "1 for limited workspace (default), 2 for fastest.");
    /*
	利用可选类cudnn_tune来选择卷积算法. 利用函数add_enum(const std::string &key, int value)来为cudnn_tune添加参数. 添加一对
	键值key和value, conv::kOff为0, conv::kLimited为1, conv::kFastest为2. 默认值是dmlc::optional<int>(), 即类optional的默认
	构造函数, 没有值为空.   
	*/
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
    .describe("Turn off cudnn for this layer."); // 是否使用cudnn, 默认是false. 即使用. 
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NCHW", mshadow::kNCHW)
    .add_enum("NHWC", mshadow::kNHWC)
    .add_enum("NCDHW", mshadow::kNCDHW)
    .add_enum("NDHWC", mshadow::kNDHWC)
    .set_default(dmlc::optional<int>())
    .describe("Set layout for input, output and weight. Empty for\n    "
              "default layout: NCHW for 2d and NCDHW for 3d.");
    /*
	设置输入, 输出和权重的layout(布局), 定义本层输入data, 输出和权重的布局, 因为输入和输出均是4D的数据, 因此要设定在内存存储,
	是以行为主函数列为主. Blob memory is row-major in layout, so the last / rightmost dimension changes fastest. 
	默认2d使用NCHW, 3d使用NCDHW.   kNCHW = 0, kNHWC, kCHWN, kNCDHW = 1 << 5, kNDHWC,
    kCDHWN是枚举变量. 定义在mshadow/mshadow/base.h下.  
	*/
  }
};

template<typename xpu, typename DType>
class ConvolutionOp : public Operator { // 卷积操作类, 前向操作和反向操作. 
 public:
  explicit ConvolutionOp(ConvolutionParam p) {
    // explicit关键字只能用于修饰只有一个参数的类构造函数, 它的作用是表明该构造函数是显示的, 而非隐式的. 参数是卷积参数p. 
    this->param_ = p;
    // p是ConvolutionParam卷积层参数类的对象, 将p赋值给param_. 这和单纯的赋值不一样, param_就是p, 可以看做是指向p的指针. 
    
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    // param_.workspace赋值, 首先令param_.workspace为20, 再除以sizeof(DType). 例如sizeof(float). 
    CHECK(param_.layout.value() == mshadow::kNCHW ||
          param_.layout.value() == mshadow::kNCDHW)
      << "Only support NCHW and NCDHW layout";
    // 卷积层的layout只支持NCHW and NCDHW, 判断卷积层的layout.value()(layout的值, 返回一个int的数)是否是mshadow::kNCHW或
	// mshadow::kNCDHW. 如果不相等, 输出断言. 
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    /*前向传播, 虚函数. 函数的实现在类中定义. 不需要返回值. 本层为第 l 层. 
	in_data: 本层输入data, 本层的权重wmat和偏置(卷积层可能有偏置, 可能没有).
	req: 数据操作模式. 
	out_data: 本层输出, out. 
	*/
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[conv::kOut], kWriteTo);
    /*
	判断卷积层前向传播的数据模式是不是kWriteTo, 如果不是, 断言. 一般情况下, 所有的out_data的类型应该是kWriteTo, 
	表示out_data代表的tensor是可以直接写入的原始的内存块 .  
	*/
    size_t expected = param_.no_bias ? 2 : 3;
    /*
	size_t是标准C库中定义的, 应为unsigned int, 在64位系统中为 long unsigned int. 定义expected, param_.no_bias为真, 则expected
	为2, 否则为3. 即expected是用来定义前向传播中的参数的个数的, 是只有kData, KWeight, 还是kData, KWeight,, kBias. 
	*/
    CHECK_EQ(in_data.size(), expected); // 检查输出in_data(向量数组)的大小是否和expected相等, 不相等, 断言. 
    CHECK_EQ(out_data.size(), 1); // 前向操作只有一个输出out, 因此前向操作中的out_data的大小应该是1.
	 
    Stream<xpu> *s = ctx.get_stream<xpu>(); // XPU流. CPU或GPU. 
    if (param_.kernel.ndim() > 2) {
      LOG(FATAL) << "Volume convolution is not implmented in mshadow";
    }
    /*
	param_就代表了卷积层的参数实例, 利用param_可以调用卷积层的任意一个参数. kernel.ndim()即TShape对象的维数, mshadow只做了二维
	卷积. convolution-inl.h(0.9版本的)做了3D卷积, 基于NNVM. 
	
	log的级别:
	error是错误; fatal是致命; silence: 显示最简练的日志信息; verbose：显示最详细的日志信息.
	*/
    Tensor<xpu, 4, DType> data = in_data[conv::kData].get<xpu, 4, DType>(s);
    /*
	将本层(第l层)的输入数据in_data[kData]拉成4维的张量. 这使用get函数:
    mshadow::Tensor<Device, dim, DType> mxnet::TBlob::get(mshadow::Stream<Device> *stream = NULL)const. 4维的张量, 这里就和
	blob比较类似了, 即number N x channel K x height H x width W, (N, K, H, W). Blob memory is row-major in layout. 
	
	这的channel K不一定是图像的通道数了, 或者为前一层特征图的数量.  
	*/
    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               param_.num_filter / param_.num_group,
               data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    /*
	在https://raw.githubusercontent.com/jdeng/gomxnet/master/mxnet.cc可以找到Shape1, Shape2, Shape3, Shape4的定义:
	Shape3定义如下:
	MSHADOW_XINLINE Shape<3> Shape3(index_t s0, index_t s1, index_t s2) {
        Shape<3> s;
        s[0] = s0; s[1] = s1; s[2] = s2;
        return s;
    }
    
    定义wmat_shape, 为定义本层(第l层)的权重wmat做准备. 即卷积层权重矩阵的shape. 
    s0 = param_.num_group, 即num_group(输入数据切割成num_group个partitions), 默认为1;
	s1 = param_.num_filter / param_.num_group. num_filter为卷积核的个数, 在使用COnvolution时指定, num_filter的范围, 1-100000.
	s2 = data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1];
	data.shape_[1]是data shape的第二个分量的值, 即channel K. 这里为卷积层前一层特征图的数量.  
    param_.kernel[0]即kernel[0], 只是使用param_来调用kernel. 在使用卷积层的时候: kernel = (5,5).  
    param_.kernel[1]即kernel[1]. 
	*/
    Tensor<xpu, 3, DType> wmat =
        in_data[conv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
    /*
	将卷积层权重in_data[conv::kWeight]即in_data[1]拉成3维的张量. 这利用了get_with_shape. 定义如下:
	mshadow::Tensor<Device, dim, DType> mxnet::TBlob::get_with_shape(const mshadow::Shape<dim> & shape, 
	mshadow::Stream< Device > *stream = NULL)const, 其中Shape<dim> & shape 就是wmat_shape. 
	
	卷积层的权重wmat是3D的张量. 权重的大小是: wmat_shape. 一般的, 卷积层的卷积核(weight)的个数为:
	num_filter * (kernel[0] * kernel[1]). 由于mxnet做卷积时, 用到了 num_group 机制, 因此将wmat设置成3D的, 其中第一个维度是
	num_group的数目. 假设要将卷积层的输入数据data分成num_group个partitions, 那么每个partitions的卷积核个数就是:
	param_.num_filter / param_.num_group个.   
	*/
    Tensor<xpu, 4, DType> out = out_data[conv::kOut].get<xpu, 4, DType>(s);
    /*定义本层(第l层)的输出数据out_data[kOut]即out_data[0], 为4维的张量.*/
    
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif // __CUDACC__ // 是否定义了__CUDA__宏. 
    const index_t nbatch = data.size(0);
    /*
    index_t的定义在mshadow/mshadow/base.h下.
	typedef unsigned index_t; 
	unsigned a; 与unsigned int a; 是同样的. 这里省略了int, 只能和unsigned int等价. 
    
	Tensor<xpu, 4, DType> data; data是4维的张量, 张量的大小是(N, C, H, W), 第一维是数据子集的个数, 可以看做是批训练中一批数据
	个数; C是图像的通道数, 即数据是二维的还是三维的数据(这的channel K不一定是图像的通道数了, 或者为前一层特征图的数量. ); 
	H是数据的高度; W是数据的宽度. 因此, data.size(0)即N, data.size(1)即C, data.size(2)即H, data.size(3)即W. 
	nbatch是索引类型(unsigned int)的变量, 代表了批量大小. 
	*/
	
    Tensor<xpu, 1, DType> workspace =
        ctx.requested[conv::kTempSpace].get_space_typed<xpu, 1, DType>(
            Shape1(this->InitTemp(data.shape_, out.shape_)), s);
	/*
	get requested temp space. 获取所需的临时空间. 
	有些操作需要额外的内存作为工作空间进行计算, 比如说cudnnConvolutionForward. 这种情况下, 系统最好可以对这部分内存进行管理, 
	这样系统可以做一些优化, 比如说内存的重复利用. 因此BN有kTempSpace. 即BN的反向操作会申请一个临时的资源空间, 这个空间任意. 
	*/            
    /*
	workspace是一维的张量(向量). 
	OpContext: 结构体, 定义在include/mxnet/operator.h中, 该结构体可以记录操作在前向和后向传播中的信息. ctx是结构体OpContext定
	义的对象, requested是OPContext结构体下的函数:
    // brief Resources requested by the operator
  	std::vector<Resource> requested; // 用来返回卷积操作所需的资源. 
    ctx.requested返回的是一个向量容器, 我们需要的只是kTempSpace即卷积层的资源配置, 即一个临时的操作空间. 
	ctx.requested[conv::kTempSpace]就是一个Resource的对象, 再调用get_space_typed函数, 将这个卷积的操作空间拉成一个一维的张量.
	
	Resource结构体定义: http://mxnet.io/doxygen/structmxnet_1_1Resource.html. 定义了mxnet操作所需的资源.
    get_space_typed函数:
    mshadow::Tensor<xpu, ndim, DType> mxnet::Resource::get_space_typed (mshadow::Shape<ndim> shape, mshadow::Stream<xpu>* 
	stream )const 返回指定类型的Tensor. 
	其中, shape是返回的Tensor的Shape, stream是所需的流. 
	
	MSHADOW_XINLINE Shape<1> Shape1(index_t s0) {
	    Shape<1> s; s[0] = s0;
        return s;
    } 
    
    这里第一个变量的定义用到了InitTemp函数, 该函数也是定义在类COnvolutionOP下:
    inline index_t InitTemp(const mshadow::Shape<4> &ishape, const mshadow::Shape<4> &oshape) {...} 
    data.shape_, out.shape_ 返回输入和输出的shape, 由于data和out均是4维的张量, 所以data.shape_, out.shape_均是4维的. 类型是
	Shape<4>, InitTemp函数返回值类型是index_t, 正和Shape1的参数类型一致. Shape1(this->InitTemp(data.shape_, out.shape_))返回
	的就是Tensor的shape. 
	*/        
            
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      
      // cout<<"step: "<<step<<endl;  // 64, ji batch_size. 
      /*
	  step是索引类型(unsigned int), index_t nstep_ 即nstep_是类ConvolutionOp下的一个数据成员, 在运行时会指定nsetp_的大小. 
	  nstep_是batch_size. 因此for循环只执行一次, 即 i == 0.  
	  step取nstep_ 和 nbatch_ - i 的最小值. 由于nstep_是batch_size, 因此step为batch_size. 
	  */
	  
      Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(workspace.dptr_,
                                               Shape2(shape_colunit_[0],
                                                      shape_colunit_[1] * step), s);
      Tensor<xpu, 3, DType> temp_dst = Tensor<xpu, 3, DType>(
                                               workspace.dptr_ + temp_col.shape_.Size(),
                                               Shape3(shape_dstunit_[0],
                                                      shape_dstunit_[1],
                                                      shape_dstunit_[2] * step), s);
	  /*
	  lenet网执行时的特征图大小:
	  28*28 --> 12*12(conv1) --> 6*6(pool1) --> 2*2(conv2) --> 1*1(pool2) -->... 
	  */
	  /*
	  这三个量shape_colunit_, shape_dstunit_, nstep_. 
  	  是在运行时指定的, 是在ConvolutionOp类的私有函数InitTemp中指定的, 最新版本的convolution-inl.h中也有一个私有函数
  	  LayerSetUp, 用来指定一些值. 根据ishape(本层输入shape)和oshape(本层输出shape)即, data.shape_, out.shape_
	  可以确定OP中用到的一些变量的值. 
	  */
	  
	  // cout<<"forward_nstep_: "<<nstep_<<endl; // nstep_是batch_size.  
	  /*cout<<"shape_colunit_[0]: "<<shape_colunit_[0]<<endl; 
	  // conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), stride=(2,2), num_filter=10)时, 为25. 
	  // shape_colunit_[0]是: 卷积层前一层特征图个数 * kernel[0] * kernel[1]. ishape[1] * ksize_y * ksize_x.
	  cout<<"shape_colunit_[1]: "<<shape_colunit_[1]<<endl;
	  // 为144, 4. 是 卷积层输出特征图的高度 * 宽度 之积. oshape[2] * oshape[3].
	  cout<<"shape_dstunit_[0]: "<<shape_dstunit_[0]<<endl;
	  //  shape_dstunit_[0]是1. param_.num_group.  
	  cout<<"shape_dstunit_[1]: "<<shape_dstunit_[1]<<endl;
	  // shape_dstunit_[1]是卷积层卷积核的个数. 为10, 50. param_.num_filter / param_.num_group.
	  cout<<"shape_dstunit_[2]: "<<shape_dstunit_[2]<<endl;
	  // 是 卷积层输出特征图的高度 * 宽度 之积.  oshape[2] * oshape[3].*/ 
	  
	  /*
	  这是利用了Tensor类的构造函数来创建xpu, ndim == 2, 3, DType下的对象temp_col和temp_dst. 需要Tensor类(结构体)的定义, 见:
	  mshadow/mshadow/tensor.h 363行. 
	  根据375和378的Tensor构造函数的形式, 确定375和378所使用的构造函数为:
	  template<typename Device, int dimension, typename DType MSHADOW_DEFAULT_DTYPE>
	  MSHADOW_XINLINE Tensor(DType *dptr, const Shape<dimension> &shape, Stream<Device> *stream) : dptr_(dptr), shape_(shape),
	  stride_(shape[kSubdim]), stream_(stream) {}. 依据data pointer and shape, without stride来构造Tensor的对象. 
	  Device会在make mxnet的指定, cpu或gpu. dimension会在定义卷积层操作的时候指定, 如2, 3. 
	  DType *dptr: 即float* 型的指针dptr;
	  Shape<dimension> &shape: 是一个shape, 可用Shape2或者Shape3定义(Shape2和Shape3定义如下). 即定义的Tensor对象的shape.  
	  Stream<Device> *stream: 是Stream对象. 
	  
	  *dptr传入的实参一个是workspace.dptr_, 一个是workspace.dptr_ + temp_col.shape_.Size().
	  workspace定义: Tensor<xpu, 1, DType> workspace, 即是1维Tensor的对象, 引用 struct Tensor<Device, 1, DType> 结构体下的
	  成员: DType *dptr_; 514行. 是指向数据的指针. 
	  temp_col.shape_.Size()是 temp_col.shape_各个分量的乘积, 即 shape_colunit_[0] * shape_colunit_[1] * step, 是一个数. 
	  
	  用了Tensor类的构造函数来创建 xpu, ndim == 2, 3, DType下的Tensor对象temp_col和temp_dst, 在创建的时候会指定
	  temp_col和temp_dst的shape. temp_col和temp_dst是两个临时的Tensor, 利用tmp_col和tmp_dst来计算最终卷积层的输出Tensor
	  Tensor<xpu, 4, DType> out.  
	  
	  Shape3定义如下:
	  MSHADOW_XINLINE Shape<3> Shape3(index_t s0, index_t s1, index_t s2) {
        Shape<3> s;
        s[0] = s0; s[1] = s1; s[2] = s2;
        return s;
  	  }
  	  
	  Shape2定义如下:
	  MSHADOW_XINLINE Shape<2> Shape2(index_t s0, index_t s1) {
        Shape<2> s;
        s[0] = s0; s[1] = s1;
        return s;
      }
	  
	  // MSHADOW_XINLINE is used for inlining template code for both CUDA and CPU code. MSHADOW_XINLINE是一个宏. 
	  #ifdef MSHADOW_XINLINE
	      #error "MSHADOW_XINLINE must not be defined"
	  #endif
	  */
 
      if (param_.pad[0] == 0 && param_.pad[1] == 0) { // 0.8的卷积操作只能实现2D卷积, 因此kernel, stride, dilate, pad均是2D的.
	    // param_指向结构体ConvolutionParam对象p的指针, 用来访问成员pad. 即pad[0], pad[1]. 
		// 做2D卷积时, 不对卷积核补零, temp_col是:
		
		/*// data是卷积层的输入数据! data是 64 * 1 * 28 * 28的. 手写数字体识别数据.  			   
		cout<<"data.shape_[0]: "<<data.shape_[0]<<endl; // 64 
		cout<<"data.shape_[1]: "<<data.shape_[1]<<endl; // 1, 通道数. channel K. 单通道(灰度图像), RGB图像(彩色图像). 
		cout<<"data.shape_[2]: "<<data.shape_[2]<<endl; // 28 
		cout<<"data.shape_[3]: "<<data.shape_[3]<<endl; // 28 */ 
		
		/*
		// 输出卷积层输入数据 data(4D tensor)的数据指针. 
		cout<<"data[0]: "<<data[0].dptr_<<endl; // data[0]: 0x7f98600008c0 
		cout<<"data[1]: "<<data[1].dptr_<<endl; // data[1]: 0x7f9860001500 
		cout<<"data[2]: "<<data[2].dptr_<<endl; // data[2]: 0x7f9860002140 
		cout<<"data[3]: "<<data[3].dptr_<<endl; // data[3]: 0x7f9860002d80 
		*/
		
		/*
 		// 写入数据, 将data数据保存到.txt中. 因为cout输出一个tensor会出错, 
		// 所以只操作单个值. 
		// std::ofstream fdata("/home/ly/MXNet/fdata.txt");
		for(int i = 0; i < data.shape_[0]; ++i){
				for(int j = 0; j < data.shape_[2]; ++j){
						for(int k = 0; k < data.shape_[3]; ++k){ 
								fdata << data[i][0][j][k] <<" "; 
						}
				}
		}
		fdata<<flush;
	 	// fdata.close();
		// data的类型是mshadow::Tensor<mshadow::cpu, 4, float>&
		// data[0]的类型是mshadow::Tensor<mshadow::cpu, 3, float>
		// data[0][0]的类型是mshadow::Tensor<mshadow::cpu, 2, float>
		// data[0][0][0]的类型是mshadow::Tensor<mshadow::cpu, 1, float>
		// data[0][0][0][0]的类型是float&. 
		*/
		
        temp_col = unpack_patch2col(data.Slice(i, i + step),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    param_.dilate[0],
                                    param_.dilate[1]);
        /*
		data.Slice(i, i + step), 用到了Slice函数, 切片函数, 将一个Tensor数据切片, 根据data.Slice(i, i + step)的调用格式, Slice
		应该是Tensor结构体的成员, slice函数和Slice函数是不一样的, Slice定义见:
		mshadow/mshadow/tensor.h 481行:
		MSHADOW_XINLINE Tensor<Device, dimension, DType> Slice(index_t begin, index_t end) const {...}. Slice是结构体Tensor下
		成员函数, 因此Tensor的对象data可以调用Slice函数.
		slice the tensor in highest dimension [begin,end)
		param begin: begin position of slice
		param end: end position of slice
		return tensor after slice. 返回切片后的Tensor. 
		*/
		
		/*
        // cout<<"i: "<<i<<endl; // 0, i一直是0, 循环仅执行1次. 
        // 输入iamge后面就是卷积层. 
		auto a = data.Slice(i, i + step); // c++11类型推断.
        cout<<"a.shape_[0]: "<<a.shape_[0]<<endl; // step, 因为step是batch_size, 所以为64. 或者a.size(0). 
        cout<<"a.shape_[1]: "<<a.shape_[1]<<endl; // 1. 
        cout<<"a.shape_[2]: "<<a.shape_[2]<<endl; // 28.
		cout<<"a.shape_[3]: "<<a.shape_[3]<<endl; // 28.
		*/
		
		/*
		由于Tensor型的变量不能直接用cout输出, 所以利用mshadow/mshadow/tensor.h 363行, Tensor结构体的成员来访问Tensor的内容:
		Tensor结构体的成员变量:
	    pointer to the data 
  		DType *dptr_; // 指向Tensor数据的指针 dptr_.  
  		
		shape of the tensor
  		Shape<dimension> shape_; // 利用 shape_ 来访问Tensor的shape. 
  		
		storing the stride information in x dimension
		this is used to deal with pitch allocation in gpu or sse(align x dimension to 64bit) for efficiency
		index_t stride_;
		
		stream where the computation lies
		stream is a device dependency concept where each computation
		Stream<Device> *stream_; 
		*/
		/*
		// 因此输出tensor时用dptr_试试.
		cout<<"a[0]: "<<a[0].dptr_<<endl; // a[0]: 0x7f98600008c0
		cout<<"a[1]: "<<a[1].dptr_<<endl; // a[1]: 0x7f9860001500 
		cout<<"a[2]: "<<a[2].dptr_<<endl; // a[2]: 0x7f9860002140 
		cout<<"a[3]: "<<a[3].dptr_<<endl; // a[3]: 0x7f9860002d80 
		*/
		/*
		实验下, 由于step = batch_size, 输出tensor的dptr_. 一个用data, 一个用data.Slice(i, i + step). 比较差异.
		根据 data 和 data.Slice(i, i + step)的大小比较 + dptr_ 比较, 由于step是batch_size, 认为卷积层的输入数据没有发生改变. 
		
		data.Slice(i, i + step)是对data这个tensor进行切片, 取data的一部分(i, i + step). 
		*/
                                    
        /*计算temp_col: 
		利用函数unpack_patch2col: 定义见mshadow/mshadow/extension/unpack_patch2col.h 91和104行. 输入参数不同, 根据445使用的
		unpack_patch2col, 函数定义为:
	    template<typename SrcExp, typename DType, int etype>
		inline UnpackPatchToColXExp<SrcExp, DType, ExpInfo<SrcExp>::kDim> unpack_patch2col(
		const Exp<SrcExp, DType, etype> &img, index_t psize_y, index_t psize_x, index_t pstride_y_, index_t pstride_x_,
		index_t pdilate_y_, index_t pdilate_x_) {...}. 
		将图像块(patches of image)unpack(解压)成一个矩阵的一列, 在利用unpack_patch2col得到mat后, 可以实现卷积.
	
		img: source image, img可以是3D Tensor或者4D Tensor(多幅图像). 这里是4DTensor, batch_size * 1 * 28 * 28. 
		psize_y: 每个patch的高度. 这里是kernl[0], 即卷积核的高度. 
		psize_x: 每个patch的宽度. 这里是kernel[1], 即卷积核的宽度. 
		pstride_y_: 每个patch在y方向上的滑动步长. 这里是stride[0], 即卷积核在y方向上的滑动步长. 
		pstride_x_: 每个patch在x方向上的滑动步长.  这里是stride[1], 即卷积核在x方向上的滑动步长. 
		pdilate_y_: 每个patch在y方向上的膨胀系数. 这里是dilate[0], 即卷积核的膨胀系数, y方向. 
		pdilate_x_: 每个patch在x方向上的膨胀系数. 这里是dilate[1], 即卷积核的膨胀系数, x方向. 
		
		利用unpack_patch2col得的mat, 得到output: output = dot(weight, mat). output即是卷积层的输出特征图. 
		out_height = [(in_height - psize_y)] / pstride + 1,
		out_width  = [(in_width - psize_x)] / pstride + 1. 
		*/
		
		/*
		cout<<"temp_col.shape_[0]: "<<temp_col.shape_[0]<<endl; // 25, param_.kernel[0] * param_.kernel[1] 
		// 这个大小是 shape_colunit_[0]是: 卷积层前一层特征图个数 * kernel[0] * kernel[1]. 
		cout<<"temp_col.shape_[1]: "<<temp_col.shape_[1]<<endl; // 9216, batch_size * [ out.size(2) * out.size(3) ], out是
		// 卷积层输出特征图.  这个大小是 shape_colunit_[1] * step, 是 step * 卷积层输出特征图的高度 * 宽度 之积.
		// 在定义张量temp_col时, 已将其大小定义好. 
		cout<<"temp_col.shape_.Size(): "<<temp_col.shape_.Size()<<endl; // 230400 = 25 * 9216.
		
		将图像块(patches of image) 28 * 28的patch, unpack(解压)成一个矩阵的一列, 
		利用unpack_patch2col得到mat后, 可以实现卷积.
		*/
		
		
      } else {
        temp_col = unpack_patch2col(pad(data.Slice(i, i + step),
                                    param_.pad[0], param_.pad[1]),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    param_.dilate[0],
                                    param_.dilate[1]);
        /*
		如果pad[0]或pad[1]不是0, 先利用 pad函数 对 data.Slice(i, i + step) 进行补零操作, 然后再做unpack_patch2col操作.
		
		pad函数也是定义下mshadow::expr命名空间下, 因此要using namespace mshadow::expr; 定义见mshadow/mshadow/extension/pad.h. 
		由于241行使用的pad有三个参数, 因此:
		template<typename SrcExp, typename DType, int etype>
		pad(const Exp<SrcExp, DType, etype> &src, index_t pad_y, index_t pad_x)
		对一张图片进行补零操作, 在图片的四周补零. src原图像; pad_y: padding size in y, 即在y方向上补零的行数; pad_x: 
		padding size in x, 在x方向上补零的列数. 返回补零的结果, 即返回补完零之后的矩阵.  
		*/                            
                                    
      } 
	  // temp_col 是做完 unpack_patch2col 后的tensor, 大小是 shape_colunit_[0] * [shape_colunit_[1] * step]. 

      const index_t gstride = temp_col.size(0) / param_.num_group; 
      // gstride类型是index_t, 即unsigned int型的. 其值为 temp_col.size(0) / param_.num_group.
	  // 即 shape_colunit_[0](卷积层前一层特征图个数 * kernel[0] * kernel[1].) / num_group(默认为1, 
	  // 将输入数据切割成num_group个partitions. num_group貌似只能是1). 
      
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) { 
	    /*
		使用typedef给类型起的别名. 1字节:uint8_t; 2字节: uint16_t; 4字节: uint32_t; 8字节: uint64_t.
		typedef unsigned char uint8_t;
		typedef unsigned int uint16_t; 
		typedef unsigned long uint32_t;
		typedef unsigned long long uint64_t;
		
		param_是ConvolutionParam结构体的对象, 调用成员num_group. 因为num_group默认为1, 因此 gid == 0. 即只做一次for循环.
		这个for循环做的就是: 将输入数据切割成num_group个partitions. 有点并行的意思. 
		*/

        mshadow::Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid,
                                       gstride * (gid + 1));
        /*
		temp_col是做完unpack_patch2col之后的Tensor, 在pad默认为0的情况下, 其大小为: 
		shape_colunit_[0] * [shape_colunit_[1] * step]. 再次调用Slice, tensor切片函数:
		Slice函数, 调用格式: data.Slice(i, i + step)的调用格式. 
		
		gid的总数是 num_group - 1. 即gid起的作用就是, 将一个tensor(输入数据切割成num_group个partitions), gid就代表每个
		partitions的索引, 即第 gid 个partitions. 这里是利用Slice函数切割temp_col, 切割的范围是 gid-gid+1, 即将temp_col切割成
		1个partitions. 每个partitions(也是一个tensor)的大小是 gstride.  
		 
		*/
		/*
		// 输出tmpc的大小, 是一个2维的张量.
		cout<<"tmpc.shape_[0]: "<<tmpc.shape_[0]<<endl; // 25 
		cout<<"tmpc.shape_[1]: "<<tmpc.shape_[1]<<endl; // 9216 
		
		在默认 num_group == 1的情况下, gid == 0. gstride == temp_col.size(0). 
		因此, 为temp_col.Slice(0, gstride). 故, tmpc 和 temp_col是一样的. 这里因为没有做 输入数据切割成num_group个partitions
		这个操作, 因此 tmpc 和 temp_col是一样的. 
		*/
		
                                       
        temp_dst[gid] = dot(wmat[gid], tmpc);
        /*
		temp_dst是3D的tensor. gid == 0, 因此是给 temp_dst[0]赋值, 根据 unpack_patch2col 函数的说明, 
		output: output = dot(weight, mat)即可以看做是卷积后的结果(离散卷积即, 卷积核元素和对应位置的元素相乘在相加, 即点乘dot
		. 不过dot现在是矩阵之间的点乘了). 
		因此, temp_dst就是做完卷积的结果. 
		
		1)num_group == 1的情况下, tmpc和tmp_col是一样大小的, 代表做完unpack_patch2col后的tensor. 即tmpc是卷积层的输入数据. 
		tmpc有两种情况, pad[0] == pad[1] == 0 或不全为0. 总之, 就是卷积层的输入数据. 
		
		2)wmat[0]就是卷积层全部的卷积核的权重值(由于权值共享, 因此一个特征图需要一个卷积核).
		wmat是卷积层的权重, 即卷积核的权重. wmat定义如下:
  		Shape<3> wmat_shape =
        Shape3(param_.num_group,
               param_.num_filter / param_.num_group,
               data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
	   Tensor<xpu, 3, DType> wmat;  
	   
	   wmat是3Dtensor, 因此wmat[0]就是2D的tensor. 因此, 默认num_group == 1, 因此wmat的第一维最大就是0. 故wmat[0]的维数是:
	   { param_.num_filter / param_.num_group } * { data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1] } 
	   即: 卷积核的个数 * channel K * kernel[0] * kernel[1]. channel K == 1是做2D卷积. 因此, num_group == 1的情况下, 
	   wmat[0]就是卷积层全部的卷积核的权重值(由于权值共享, 因此一个特征图需要一个卷积核). 
	   批处理中, 对于一个样本来说, 卷积层的权重是相同的. 
	   
	   3)temp_dst[0] 即是对卷积层输入数据 tmpc 做完卷积(仅weight)后的 输出数据(批处理的).
	   根据unpack_patch2col 函数的说明, dot(weight, tmpc)就是做完卷积后的结果. 
	   temp_dst[0]就是做完卷积后的结果, 这里仅仅和卷积核发生了作用. 还没有涉及偏置.
	   tem_dst定义如下:
	   Tensor<xpu, 3, DType> temp_dst; 大小为: shape_dstunit_[0] * shape_dstunit_[1] * { shape_dstunit_[2] * step), s) };
	   shape_dstunit_[0]是1, 即channel K; shape_dstunit_[1]是卷积层卷积核的个数; shape_dstunit_[2]是 积层输出特征图的
	   高度 * 宽度 之积. 因此, temp_dst的第一维最大值为0. 
	   即num_group == 1的情况下, temp_dst[0]是2D的tensor, 大小是:
	   卷积核的个数(卷积层输出的特征图个数) * { 特征图大小 * batch_size }. 批处理中, 一个样本经过卷积层后的输出数据大小是:
	   卷积核的个数(卷积层输出的特征图个数) * 特征图大小; 对于不同的样本, 这个大小相同, 只是值不同. 
	   因此, temp_dst[0] 即是对卷积层输入数据 tmpc 做完卷积(仅weight)后的 输出数据(批处理的).   
		*/
      }
      // 以上的有些参数的值的结果是在 num_group == 1和 step == batch_size的情况下, 推出来的. 如wmat的大小, wmat[gid]的大小, 
	  //tmpc的大小, tmp_col的大小, tmp_dst的大小等等. 這下tensor在 num_group != 1時, 均要發生变化. 这里不再细研究了.   
      out.Slice(i, i + step) = swapaxis<1, 0>(reshape(temp_dst,
                                              mshadow::Shape4(param_.num_filter,
                                                  step,
                                                  out.size(2),
                                                  out.size(3))));
	  /*
	  num_group == 1的情况下, gid == 0. temp_dst[0] 即是对卷积层输入数据 tmpc 做完卷积(仅weight)后的输出数据(批处理的). 大小是:
	  卷积核的个数(卷积层输出的特征图个数) * { 特征图大小 * batch_size }.
	  
	  // cout<<"i: "<<i<<endl; // 0, i一直是0, 循环仅执行1次. 
	  // 输入iamge后面就是卷积层. 
	  auto a = data.Slice(i, i + step); // c++11类型推断.
	  cout<<"a.shape_[0]: "<<a.shape_[0]<<endl; // step, 因为step是batch_size, 所以为64. 或者a.size(0). 和index_t nstep_;相关. 
	  cout<<"a.shape_[1]: "<<a.shape_[1]<<endl; // 1. 
	  cout<<"a.shape_[2]: "<<a.shape_[2]<<endl; // 28.
	  cout<<"a.shape_[3]: "<<a.shape_[3]<<endl; // 28.
	  data是卷积层的输入数据, 还未做 unpack_patch2col操作. data.Slice(i, i + step)即将data切片成 [nbatcb/nstep_]份, 并行处理.
	  
	  如果卷积层的输入数据data别切成了 [nbatcb/nstep_]份, 做并行处理, 那么 tmp_col 也是 [nbatcb/nstep_]份, tmp_dst也是
	  [nbatcb/nstep_]份. 因此, temp_dst[gid] = dot(wmat[gid], tmpc)(假设gid==0)就是做的是一份data的卷积后的结果. 
	  out是卷积层的输入tensor, 是4D的. 如果 tmp_dst也是[nbatcb/nstep_]份的, 那么out也别切成了 [nbatcb/nstep_]份. 因此要一份份
	  的赋值, 最终的输出是out. 
	  即 out.Slice(i, i + step), 这是对卷积层的输出out(一份, 即大小为step)的赋值操作. 将[nbatcb/nstep_]份out赋值完毕才算是得到
	  了卷积层的真正输出. 再将一份 temp_dst[0] 赋给一份的 out时, 要改变大小. 因为out是4D的, temp_dst[0]是2D的tensor.
	  
	  1)reshape函数, 重新定义输入的大小. 这个reshape不是python numpy中的reshape, reshape输入接收的第一个参数是tmp_dst, 类型是
	  Tensor. 定义见: mshadow/mshadow/extension/reshape.h 48行. 在mshadow::exprc命名空间下. 
	  template<typename SrcExp, typename DType, int etype, int dimdst>
	  inline ReshapeExp<SrcExp, DType, dimdst, ExpInfo<SrcExp>::kDim> reshape(const Exp<SrcExp, DType, etype> &src, 
	  Shape<dimdst> oshape) {...}. reshape a tensor to another shape.
	  src: Tensor<Device, dimsrc>, dimsrc是src的维数, 如3.
	  oshape: target shape. 是Shape<dimdst> oshape, dimdst是输出tensor的维数, 如4.
	  return a expresion with type Tensor<Device, dimdst>. 
	  因此reshape(temp_dst, *)就是重新temp_dst这个3D Tensor的shape, 但是数据的总个数是不变的. temp_dst[0]是2D的tensor, 大小是:
	  卷积核的个数(卷积层输出的特征图个数) * { 特征图大小 * batch_size }. 因此, temp_dst的大小为:
	  channel K * 卷积核的个数(卷积层输出的特征图个数) * { 特征图大小 * batch_size }.
	  reshape函数输出的shape为: num_filter * step * out.size(2) * out.size(3). 即:
	  卷积核个数 * step(batch_size) * 卷积层输出特征图高度 * 宽度(4D). 这样正好对上卷积层输出out tensor的维数.
	  
	  reshape(temp_dst, *)就返回 卷积核个数 * step(batch_size) * 卷积层输出特征图高度 * 宽度(4D)的一个4Dtensor.
	  
	  2)swapaxis函数, 函数定义见mshadow/mshadow/extension/swapaxis.h 52行. 在mshadow::exprc命名空间下. 
	  template<int a1, int a2, typename SrcExp, typename DType, int etype>
	  inline SwapAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim, ExpInfo<SrcExp>::kDim - a1, a2> swapaxis(
	  const Exp<SrcExp, DType, etype> &src) {...}. reshapes a tensor to another shape.
	  src: Tensor<Device, dimsrc>, dimsrc是src的维数, 如4. 
	  return a expresion with type Tensor<Device,dimdst>. dimdst是输出tensor的维数, 如4.
	  模板参数a1: higher dimension to be swapped, assert a1 > a2. assert宏的原型定义在<assert.h>中, 
	  其作用是如果它的条件返回错误, 则终止程序执行.
	  模板参数a2: lower dimension to be swapped. 
	  
	  swapaxis<1, 0>(...)就是对 reshape(temp_dst, *)的返回结果再重新reshape一下(..). 1, 0即指定模板参数:
	  int a1, int a2. 模板类, 模板函数的模板参数可以这样指定: swapaxis<1, 0>(...).  
	  
	  最后将reshape的结果(一份temp_dst)赋给一份 out. 一共[nbatcb/nstep_]份. 
	  
	  */
	  
    }
    if (!param_.no_bias) { // no_bias默认为false, 这个if即使用bias. 卷积层默认不使用bias(偏置). 
      // add bias, broadcast bias to dim 1: channel
      Tensor<xpu, 1, DType> bias = in_data[conv::kBias].get<xpu, 1, DType>(s);
      // 利用get函数将in_data[2]拉成1维的张量, 即向量. 即卷积层的如果有bias, 其是向量. bias是一个1D的tensor. 
      // cout<<"bias.size(0): "<<bias.size(0)<<endl; // 输出卷积层bias的大小, 1D的tensor的大小. 为卷积层卷积核的个数.
	  // 即一个卷积核对应一个bias, 也是共享的. 
      out += broadcast<1>(bias, out.shape_);
      /*
	  broadcast见: mshadow/mshadow/extension/broadcast.h 69行:
	  template<int dimcast, typename SrcExp, typename DType, int etype, int dimdst>
	  inline Broadcast1DExp<SrcExp, DType, dimdst, dimdst - dimcast> broadcast(const expr::Exp<SrcExp, DType, etype> &src, 
	  Shape<dimdst> shape) {..}. 
	  src Tensor<Device,1>; shape: shape of output; 返回 a expresion with type Tensor<Device, dimdst>, dimdst为4, 
	  返回的Tensor的维数为4, 和shape的个数是有关的.
	  * input: Tensor<Device,1>: ishape[0]
	  * output: Tensor<Device,dimdst> : oshape[dimcast] = ishape[0].
	  模板参数tparam dimcast: target dimension where the 1D tensor will be broadcasted 
	  将一个1维的 Tensor 扩充成 dimdst 维的 Tensor. 为了正确计算!! 
	  
	  out.shape_是卷积层输出out的shape, 是一个Shape<4>的变量, 大小为 卷积核数 * batch_size * 特征图高度 * 宽度.
	  broadcast<1>(bias, out.shape_)就是将bias这个1D张量扩展成和卷积层输出 out, 具有一样大小的tensor. 方便做加法.
	  
	  即为out的每一个数据点都加上一个bias, 所以要将bias扩展成和out一样大小的tensor才可以做加法. bias的大小是卷积核的个数, 即
	  一个卷积核对应一个bias. 也就是卷积层的一个输出特征图对应于一个bias. 
	  */
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    /*卷积层(第l层)有参数weight和bias(可能使用偏置, 可能不适应. 如果使用偏置, 每个卷积核配备一个), 
	因此要计算的是损失J关在BN层(第l层)的残差, weight的梯度和bias的梯度. 
    !!!!!!!!!!!!!!!!梯度可以看做是损失J关于层参数的导数, 残差可以看做是损失J关于层输入的导数!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
	 
    in_grad输出残差/梯度参数, 向量容器, 每个元素的类型是TBlob. 卷积层层(第l层)的.
	out_grad输入残差/梯度参数, 向量容器, 每个元素的类型是TBlob. 上一层(第l + 1层)的残差/梯度, 计算本层的残差/梯度. 
	利用上一层的残差来计算出损失关于本层输入的残差, 从而再计算出损失关于本层参数的梯度, 再进行sgd更新. 用到的一般均是上一层的
	残差.
	 
	in_data输入参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的输入.  
	out_data输出参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的输出. 	
	req: 数据操作模式, 向量数组. 元素类型是OpReqType.
	aux_args: 表示的是为了方便计算而需要的附加的 tensor. 附加的Tensor有两个: kMovingMean, kMovingVar. 以前看的操作均没使用
	aux_args来辅助计算.
	*/
    
    using namespace mshadow;
    using namespace mshadow::expr; // 一些表达式所在的命名空间. mshadow::expr.  
    // TODO(bing): check the BLAS Handle, be careful
    if (param_.kernel.ndim() > 2) {
      LOG(FATAL) << "Volume convolution is not implmented in mshadow";
    } // 0.8版本的mxnet只能做2D卷积, 因此kernel的维数如果 > 2, 就输出一个log信息; 0.9版本的mxnet可以做3D卷积了.  
    CHECK_EQ(out_grad.size(), 1); // 卷积层的上一层(第l + 1层)传递给卷积层的只有残差, 因此out_grad容器的大小为1. 
    size_t expected = param_.no_bias == 0 ? 3 : 2; //  定义expected, 如果卷积层没有偏置就是2, 否则为3. 即用expected来控制卷积
	// 的in_data和in_grad容器的大小. 
    CHECK(in_data.size() == expected && in_grad.size() == expected); // 卷积层输入in_data和in_grad容器的大小为expected.
	// 如果卷积层有偏置, 那么in_data容器大小就是3, 包含输入kData, 卷积核权重kWeight, 偏置kBias.
	// in_grad容器的大小也是3, 包括损失关于卷积层输入的残差out, 损失关于卷积核权重的梯度, 损失关于偏置的梯度. 
	 
    CHECK_EQ(req.size(), expected); // rep容器的大小是expected. 即, 对于:
	// 损失关于卷积层输入的残差out, 损失关于卷积核权重的梯度, 损失关于偏置的梯度, 采用不同的数据操作模式. 
    CHECK_EQ(in_data[conv::kWeight].CheckContiguous(), true);
	/*
	in_data[1].CheckContiguous(), 利用CheckContiguous函数, 定义见: include/mxnet／tensor_blob.h 136行:
	inline bool CheckContiguous(void) const {
	    return shape_[shape_.ndim() - 1] == stride_;
	} CheckContiguous函数是TBlob类下的成员函数, 因此可以利用TBlob的对象调用. return whether the tensor's memory is continuous
	用来检查一个tensor的内存是不是连续的. 如是是返回true, 不是返回false.
	
	in_data[conv::kWeight]即in_data[1], 是TBlob的对象, 因此可以调用CheckContiguous函数. 用来检查in_data[1]这个tensor的内存
	是不是连续的. 
	*/
	 
    // get data
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data = in_data[conv::kData].get<xpu, 4, DType>(s);
    // 利用get函数将卷积层的输入in_data[0]拉成4D的tensor, data. 代表卷积层的输入. batch_size * channel K * 高度 * 宽度. 
    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               param_.num_filter / param_.num_group,
               data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    /*
    定义wmat_shape, 为定义本层(第l层)的权重wmat做准备. 即卷积层权重矩阵的shape. 
    s0 = param_.num_group, 即num_group(输入数据切割成num_group个partitions), 默认为1;
	s1 = param_.num_filter / param_.num_group. num_filter为卷积核的个数, 在使用COnvolution时指定, num_filter的范围, 1-100000.
	s2 = data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1];
	data.shape_[1]是data shape的第二个分量的值, 即channel K. 这里为卷积层前一层特征图的数量.  
    param_.kernel[0]即kernel[0], 只是使用param_来调用kernel. 在使用卷积层的时候: kernel = (5,5).  
    param_.kernel[1]即kernel[1]. 
	*/           
               
    Tensor<xpu, 3, DType> wmat =
        in_data[conv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
    /*
	将卷积层权重in_data[conv::kWeight]即in_data[1]拉成3维的张量. 这利用了get_with_shape. 定义如下:
	mshadow::Tensor<Device, dim, DType> mxnet::TBlob::get_with_shape(const mshadow::Shape<dim> & shape, 
	mshadow::Stream< Device > *stream = NULL)const, 其中Shape<dim> & shape 就是wmat_shape. 
	
	卷积层的权重wmat是3D的张量. 权重的大小是: wmat_shape. 一般的, 卷积层的卷积核(weight)的个数为:
	num_filter * (kernel[0] * kernel[1]). 由于mxnet做卷积时, 用到了 num_group 机制, 因此将wmat设置成3D的, 其中第一个维度是
	num_group的数目. 假设要将卷积层的输入数据data分成num_group个partitions, 那么每个partitions的卷积核个数就是:
	param_.num_filter / param_.num_group个.   
	*/    
        
    Tensor<xpu, 4, DType> grad = out_grad[conv::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> gdata = in_grad[conv::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 3, DType> gwmat =
        in_grad[conv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
    /*
	利用get函数定义卷积层:
	上一层(第l + 1层)的残差out_grad[conv::kOut]即out_grad[0], 拉成4D张量.  
	
	定义损失关于卷积层输入的残差gdata, 将in_grad[conv::kData]即in_grad[0]拉成4D的张量. mxnet在进行层的计算时, 会事先分配好内存, 
	即in_grad是卷积层的残差/梯度的容器, 要事先分配好这些内存.
	
	定义损失关于卷积层权重weight的梯度gwmat, 利用get_with_shape将in_grad[conv::kWeight]即in_grad[1]拉成3D张量. 
	其中gwmat的shape和卷积层的卷积核的权重wmat的shape是一样的.  
	
	在使用函数get或get_with_shape时, 一并将模板参数传入. 
	*/
	    
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif // __CUDACC__宏. 
    const index_t nbatch = data.size(0); // nbatch即batch_size. data.size(0)即data的第一维的大小. 
    Tensor<xpu, 1, DType> workspace =
        ctx.requested[conv::kTempSpace].get_space_typed<xpu, 1, DType>(
            Shape1(this->InitTemp(data.shape_, grad.shape_)), s);
    /*
	与卷积层前向传播中的处理是一样的. 定义一个临时内存空间workspace, 是1D的tensor. 操作需要额外的内存作为工作空间进行计算.
	InitTemp输入参数, 一个是data.shape_, 一个是grad.shape_. 
	*/
            
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      // 在实际运行过程中, nbatch == batch_size, nstep_ == batch_size. 因此step == batch_size. 这个和卷积层前向传播是一样的
	  // 设置, 即在前向传播的过程中, 根据输入数据data, weight来计算输出out. 是一份一份的进行计算的(当nstep_ < nbatch时). 
	  // 因此, 反向传播时, 在计算残差和梯度的时候也是一份一份的计算的. 也有点并行的意思.   
      Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(workspace.dptr_,
                                               Shape2(shape_colunit_[0],
                                                      shape_colunit_[1] * step), s);
      Tensor<xpu, 3, DType> temp_dst = Tensor<xpu, 3, DType>(
                                               workspace.dptr_ + temp_col.shape_.Size(),
                                               Shape3(shape_dstunit_[0],
                                                      shape_dstunit_[1],
                                                      shape_dstunit_[2] * step), s);
	  /*
	  这三个量shape_colunit_, shape_dstunit_, nstep_. 
  	  是在运行时指定的, 是在ConvolutionOp类的私有函数InitTemp中指定的, 最新版本的convolution-inl.h中也有一个私有函数
  	  LayerSetUp, 用来指定一些值. 根据ishape(本层输入shape)和oshape(本层输出shape)即, data.shape_, out.shape_
	  可以确定OP中用到的一些变量的值. 
	  */
	  
	  // cout<<"forward_nstep_: "<<nstep_<<endl; // nstep_是batch_size.  
	  /*cout<<"shape_colunit_[0]: "<<shape_colunit_[0]<<endl; 
	  // conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), stride=(2,2), num_filter=10)时, 为25. 
	  // shape_colunit_[0]是: 卷积层前一层特征图个数 * kernel[0] * kernel[1]. ishape[1] * ksize_y * ksize_x.
	  cout<<"shape_colunit_[1]: "<<shape_colunit_[1]<<endl;
	  // 为144, 4. 是 卷积层输出特征图的高度 * 宽度 之积. oshape[2] * oshape[3].
	  cout<<"shape_dstunit_[0]: "<<shape_dstunit_[0]<<endl;
	  //  shape_dstunit_[0]是1. param_.num_group.  
	  cout<<"shape_dstunit_[1]: "<<shape_dstunit_[1]<<endl;
	  // shape_dstunit_[1]是卷积层卷积核的个数. 为10, 50. param_.num_filter / param_.num_group.
	  cout<<"shape_dstunit_[2]: "<<shape_dstunit_[2]<<endl;
	  // 是 卷积层输出特征图的高度 * 宽度 之积.  oshape[2] * oshape[3].*/ 
	  /*
	  和卷积层前行传播的变量一样, 利用Tensor结构体的构造函数: Tensor<xpu, 2/3, DType>(..)来创建Tensor对象temp_col和temp_dst.
	  temp_col是2D的张量, temp_dst是3D的张量. 
	  
	  在num_group == 1的情况下: 
	  temp_col的大小为: shape_colunit_[0] * (shape_colunit_[1] * step), 即: 
	  { 卷积层前一层特征图个数 * kernel[0] * kernel[1] } * { 卷积层输出特征图的高度 * 宽度 * step }. 
	  temp_dst大小为: shape_dstunit_[0] * shape_dstunit_[1] * (shape_dstunit_[2] * step), 即:
	  { channel K } * { 卷积层卷积核的个数 } * { 卷积层输出特征图的高度 * 宽度 之积 }.
	  */
                                                      
      temp_dst = reshape(swapaxis<1, 0>(grad.Slice(i, i + step)), temp_dst.shape_);
      /*
	  temp_dst是3D的tensor, shape上文已经定义了. grad是上一层(第l + 1层)的残差, 是4D的tensor.  
	  利用上一层(第l + 1层)的残差定义temp_dst, 即temp_dst就代表上一层的残差. 只是在赋值的时候需要reshape.
	  
	  1)reshape函数. 定义见: mshadow/mshadow/extension/reshape.h 48行. 在mshadow::exprc命名空间下. 
	  template<typename SrcExp, typename DType, int etype, int dimdst>
	  inline ReshapeExp<SrcExp, DType, dimdst, ExpInfo<SrcExp>::kDim> reshape(const Exp<SrcExp, DType, etype> &src, 
	  Shape<dimdst> oshape) {...}. reshape a tensor to another shape.
	  src: Tensor<Device, dimsrc>, dimsrc是src的维数, 如4.
	  oshape: target shape. 是Shape<dimdst> oshape, dimdst是输出tensor的维数, 如3.
	  return a expresion with type Tensor<Device, dimdst>.  
	  
	  2)swapaxis函数. 函数定义见mshadow/mshadow/extension/swapaxis.h 52行. 在mshadow::exprc命名空间下. 
	  template<int a1, int a2, typename SrcExp, typename DType, int etype>
	  inline SwapAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim, ExpInfo<SrcExp>::kDim - a1, a2> swapaxis(
	  const Exp<SrcExp, DType, etype> &src) {...}. reshapes a tensor to another shape.
	  src: Tensor<Device, dimsrc>, dimsrc是src的维数, 如4. 
	  return a expresion with type Tensor<Device,dimdst>. dimdst是输出tensor的维数, 如4.
	  模板参数a1: higher dimension to be swapped, assert a1 > a2. assert宏的原型定义在<assert.h>中, 
	  其作用是如果它的条件返回错误, 则终止程序执行.
	  模板参数a2: lower dimension to be swapped. 
	  
	  reshape的操作src相当于是: grad.Slice(i, i + step). 即对上一层的残差进行切片, 一份一份的来做. 一次for循环, 取一份(一共
	  [nbatch/nstep_]份)上层的残差, 来计算卷积层的残差, 权重梯度, 偏置梯度.
	  由于nbatch == nstep_, 因此 grad.Slice(i, i + step) 和 grad是相同的. i始终为0. 即拿上一层全部的残差grad来计算卷积层的
	  损失关于卷积层输入的残差, 损失关于卷积层卷积核weight的梯度, 损失关于卷积核偏置的梯度. 
	  */
      
      if (param_.pad[0] == 0 && param_.pad[1] == 0) { // pad[0] == pad[1] == 0, 即不对数据进行补零操作. 
        temp_col = unpack_patch2col(data.Slice(i, i + step),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     param_.dilate[0],
                                     param_.dilate[1]);
        // 定义temp_col, temp_col是2D的tensor. 还是利用unpack_patch2col来定义的. 
		// 由于i == 0, step == batch_size, 因此data和 data.Slice(i, i + step)是相同的. 
		// 将图像块(patches of image)unpack(解压)成一个矩阵的一列, 在利用unpack_patch2col得到mat后, 可以实现卷积前向和反向. 
      } else {
        temp_col = unpack_patch2col(pad(data.Slice(i, i + step), param_.pad[0], param_.pad[1]),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     param_.dilate[0],
                                     param_.dilate[1]);
        // pad[0]或pad[1]有一个不是0, 就先利用pad函数对data.Slice(i, i + step)进行补零操作, 然后再使用unpack_patch2col函数. 
      } // 这和前向操作的处理是一样的. data是卷积层的输入数据, 利用unpack_patch2col函数将数据数据进行unpack, 得到temp_col,
	  // 再来实现卷积层的前向和反向操作.
	    
      const index_t gstride = temp_col.size(0) / param_.num_group;
      // gstride类型是index_t, 即unsigned int型的. 其值为 temp_col.size(0) / param_.num_group.
   	  // 即 shape_colunit_[0](卷积层前一层特征图个数 * kernel[0] * kernel[1].) / num_group(默认为1, 
   	  // 将输入数据切割成num_group个partitions. num_group貌似只能是1).
		  
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) { // 与前向的操作是一样的. gid == 0, 开始循环. 一直到num_group.
	  	// 一个一个partitions的操作. num_group默认为1, 即不切割数据. 
        Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
        /*
        定义tmpc, 和前向传播一样. 是一个临时的tensor, 如果num_group != 1, 那么就将temp_col切割. 
		在默认 num_group == 1的情况下, gid == 0. gstride == temp_col.size(0). 
		因此, 为temp_col.Slice(0, gstride). 故, tmpc 和 temp_col是一样的. 这里因为没有做 输入数据切割成num_group个partitions
		这个操作, 因此 tmpc 和 temp_col是一样的. 
		
		gid起的作用就是, 将一个tensor(输入数据切割成num_group个partitions), gid就代表每个
		partitions的索引, 即第 gid 个partitions. 这里是利用Slice函数切割temp_col, 切割的范围是 gid-gid+1, 即将temp_col切割成
		1个partitions. 每个partitions(也是一个tensor)的大小是 gstride.
		*/
        
        if (i == 0) { // 因为 nbatch == nstep_, 因此 i == 0, 始终为0. 
          // cout<<"backward_i: "<<i<<endl; // 0 
          Tensor<xpu, 2, DType> tmp_gwmat = gwmat[gid]; // 可以看做是深拷贝. 
          /*
		  在默认num_group == 1的情况下, gid是0. gwmat[gid]即gwmat[0], gid就是gwmat这个tensor第一维.
		  gwmat是损失关于卷积层权重weight的梯度, 是3D的tensor, 其shape和wmat(卷积层的卷积核的权重)shape一样:
		  { param_.num_group } * { param_.num_filter / param_.num_group } 
		  * { data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1] }.
		  
		  在默认num_group == 1的情况下, gwmat这个3D的tensor, 其第一维的最大值是0. gwmat[gid]是一个2D的tensor, 在0.8版本的mxnet
		  中, 只能实现2D卷积, 因此特征图的channel K == 1. 故 tmp_gwmat这个2D tensor的大小就是:
		  num_filter * { kernel[0] * kernel[1] }. 即正好是卷积层卷积核的参数的个数. 也是损失关于卷积层卷积核参数weight的梯度
		  的个数.
		  
		  这里是定义tmp_gwmat, 即损失关于卷积层卷积核参数weight的梯度. 
		  */
          
          Assign(tmp_gwmat, req[conv::kWeight], dot(temp_dst[gid], tmpc.T()));
          /*
		  Assign赋值操作. 输出是tmp_gwmat, 即给tmp_gwmat赋值; 数据操作模式是req[conv::kWeight], 即卷积层权重的数据操作模式;
		  exp是 dot(temp_dst[gid], tmpc.T()).
		  根据 unpack_patch2col 函数的说明, output: output = dot(weight, mat)即可以看做是卷积后的结果
		  (离散卷积即, 卷积核元素和对应位置的元素相乘在相加, 即点乘dot. 不过dot现在是矩阵之间的点乘了). 
		  
		  dot(temp_dst[gid], tmpc.T())即取了转置(即: A'*B 和 B'*A正好差了一个转置). 
		  
		  其中temp_dst[gid], 在默认num_group == 1的情况下, 其为temp_dst[0]. temp_dst是3D的tensor, 其值是上一层的残差(经过了
		  reshape操作). temp_dst这个3D的tensor, 其第一维的大小是channel K. 在0.8版本的为1, 因此最大即temp_dst[0]. 
		  temp_dst[0]是一个2D的tensor, 由于 i == 0, grad.Slice(i, i + step) 和 grad是相同的, 
		  因此这个2D的tensor就是上一层全部的残差.
		  
		  tmpc: 在默认 num_group == 1的情况下, gid == 0. gstride == temp_col.size(0). 
		  因此, 为temp_col.Slice(0, gstride). 故, tmpc 和 temp_col是一样的. 这里因为没有做 输入数据切割成num_group个partitions
		  这个操作, 因此 tmpc 和 temp_col是一样的. tmpc即输卷积层的输入data经过unpack_patch2col操作后的结果, 即可以看做是卷积
		  层的真正输入数据!
		  
		  tmp_gwmat就是损失关于卷积层卷积核的参数weight的梯度:
		  其值为: 卷积层的输入 * 上一层的残差.    
		  */
          
        } else { // 如果i != 0, 那么就累加即可. 每次累加的值是: dot(temp_dst[gid], tmpc.T()). 
          gwmat[gid] += dot(temp_dst[gid], tmpc.T());
        }
      }

      for (uint32_t gid = 0; gid < param_.num_group; ++gid) { // gid循环, gid == 0. 只做一次for循环. 
        Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
        tmpc = dot(wmat[gid].T(), temp_dst[gid]);
        /*
		定义一个2D的tensor, tmpc. 定义的时候和上面的tmpc定义一样, 但是其值发生改变.
		tmpc这个2D的tensor的值为: dot(wmat[gid].T(), temp_dst[gid]). 即矩阵做点乘. tmpc在上面定义时, 是看做卷积层真正的输入. 
		
		num_group == 1的情况下, gid == 0. 
		1)wmat[0]就是卷积层全部的卷积核的权重值(由于权值共享, 因此一个特征图需要一个卷积核). 
		wmat是3Dtensor, 因此wmat[0]就是2D的tensor. 因此, 默认num_group == 1, 因此wmat的第一维最大就是0. num_group == 1的情况下, 
	 	wmat[0]就是 卷积层全部的卷积核的权重值(由于权值共享, 因此一个特征图需要一个卷积核). 
	   	批处理中, 对于一个样本来说, 卷积层的权重是相同的. 
		wmat[gid].T()取转置, T是Tensor结构体下的成员函数, 用来取一个tensor的转置.   
		
		2)其中temp_dst[gid], 在默认num_group == 1的情况下, 其为temp_dst[0]. temp_dst是3D的tensor, 其值是上一层的残差(经过了
		reshape操作). temp_dst这个3D的tensor, 其第一维的大小是channel K. 在0.8版本的为1, 因此最大即temp_dst[0]. 
		temp_dst[0]是一个2D的tensor, 由于 i == 0, grad.Slice(i, i + step) 和 grad是相同的, 
		因此这个2D的tensor就是上一层全部的残差. 
		
		tmpc就是: 卷积层卷积核的全部weight * 上一层的残差. 
		*/
      }
      
      // 计算损失关于卷积层的输入的残差, gdata. 分两种情况, pad[0]==pad[1]==0和不全为0. 
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        Assign(gdata.Slice(i, i + step), req[conv::kData],
               pack_col2patch(temp_col,
                              data.Slice(i, i + step).shape_,
                              param_.kernel[0],
                              param_.kernel[1],
                              param_.stride[0],
                              param_.stride[1],
                              param_.dilate[0],
                              param_.dilate[1]));
        /*
		计算损失关于卷积层的输入的残差, gdata. pad[0]==pad[1]==0的情况下.
		Assign赋值操作, 赋值的对象是: gdata.Slice(i, i + step), 这和卷积的前向操作是类似的. 如果卷积层的输入数据data别切成了
	    [nbatcb/nstep_]份. 因此要一份份的赋值; 数据操作模式是req[conv::kData], 即卷积层输入数据的操作模式; exp为:
		pack_col2patch(...)的返回值. unpack_patch2col函数是将卷积层的输入data进行unpack, 以便进行卷积操作; 而pack_col2patch正
		好相反, 为了得到残差服务.
		
		pack_col2patch函数见: mshadow/mshadow/extension/pack_col2patch.h 72行和88行. 有不同的参数. 根据所选参数:
		template<typename SrcExp, typename DType, int dstdim, int etype>
		inline PackColToPatchXExp<SrcExp, DType, dstdim> pack_col2patch(const expr::Exp<SrcExp, DType, etype> &src,
		Shape<dstdim> imshape, index_t psize_y, index_t psize_x, index_t pstride_y, index_t pstride_x,
		index_t pdilate_y, index_t pdilate_x) {..}. pack_col2patch是反向操作, 可以用来做反卷积(deconvolution)(mxnet也将反卷积
		deconvolution做成了单独的一层). Deconvolution是将Convolution的方向反过来--前向变后向, 后向变前向. 
		返回pack 的图像. 可以这样来看, unpack_patch2col函数是将特征图变成tensor; pack_col2patch是将tensor变成特征图.  
	
		mat: source matrix; 源矩阵为 temp_col, 即对卷积层的输入data进行unpack_patch2col后的结果, 可以看做是卷积层真正的输入.  
		imshape: 目标img的shape; data.Slice(i, i + step).shape_, 即一份data的shape, 因为i == 0, step == batch_size, 所以就是
		data.shape_, 即卷积层输入data的shape. 
		psize_y: 每个patch的高度;
		psize_x: 每个patch的宽度;
		pstride_y: 每个patch在y方向上的滑动步长;
		pstride_x: 每个patch在x方向上的滑动步长;	 
		pdilate_y_: 每个patch在y方向上的膨胀系数. 这里是dilate[0], 即卷积核的膨胀系数, y方向. 
		pdilate_x_: 每个patch在x方向上的膨胀系数. 这里是dilate[1], 即卷积核的膨胀系数, x方向. 
	  	
	  	!!!!!!!不知道在干什么!!!!!!! 
		*/
		
		// cout<<"gdata[0][0].shape_[0]: "<<gdata[0][0].shape_[0]<<endl; // 第l - 1层特征图的大小. 
		// cout<<"gdata[0][0].shape_[1]: "<<gdata[0][0].shape_[1]<<endl;
                              
      } else {
        Shape<4> pshape = data.Slice(i, i + step).shape_;
        pshape[2] += 2 * param_.pad[0];
        pshape[3] += 2 * param_.pad[1];
        Assign(gdata.Slice(i, i + step), req[conv::kData],
               crop(pack_col2patch(temp_col,
                                   pshape,
                                   param_.kernel[0],
                                   param_.kernel[1],
                                   param_.stride[0],
                                   param_.stride[1],
                                   param_.dilate[0],
                                   param_.dilate[1]),
                    gdata[i][0].shape_));
		/*
		pad[0]或pad[1]有一个不是0. 要先定义 imshape: 目标img的shape. 
		首先定义pshape, 其为: data.Slice(i, i + step).shape_. 由于pad[0]或pad[1]不为0, 因此对pshape[2]和pshape[3]进行加和:
		pshape[2] += 2 * param_.pad[0];
        pshape[3] += 2 * param_.pad[1]; // 即将pshape[2]加上 2 * pad[0]. 这样定义完的pshape就是 imshape: 目标img的shape.
		
		然后再次利用pack_col2patch函数, 操作temp_col, imshape: 目标img的shape为pshape(已经将pad信息包含在里面了).
		
		在对pack_col2patch(...)使用crop函数, 进行裁剪. 裁剪成和 gdata[i][0].shape_即gdata[0][0].shape_一样大小的tensor.
		
		gdata是4D的tensor, 因此gdata[0][0]就是2D的tensor, gdata[0][0].shape_就是损失关于卷积层输入的残差的shape. 
		是第l - 1层特征图的大小.  
		*/
		// cout<<"gdata[0][0].shape_[0]: "<<gdata[0][0].shape_[0]<<endl; // 第l - 1层特征图的大小. 
		// cout<<"gdata[0][0].shape_[1]: "<<gdata[0][0].shape_[1]<<endl; 
      }
    }
    
    /*
    cout<<"grad.size(0): "<<grad.size(0)<<endl; // 64, batch_size 
	cout<<"grad.size(1): "<<grad.size(1)<<endl; // 10, 卷积核个数. 
	cout<<"grad.size(2): "<<grad.size(2)<<endl; // 12, 卷积层输入特征图高度. 
	cout<<"grad.size(3): "<<grad.size(3)<<endl; // 12, 卷积层输出特征图宽度.
	
	即网络某层的残差要和该层的输入大小一致!! 以前也是这么做的. grad是上一层(第l + 1层)的残差. 
	*/ 
	
    if (!param_.no_bias) { // 卷积层使用偏置, 要求损失关于卷积层偏置的梯度.    
      Tensor<xpu, 1, DType> gbias = in_grad[conv::kBias].get<xpu, 1, DType>(s);
      // 因为偏置是1D的tensor, 所以其梯度也是1D的. 利用get函数将卷积层的in_grad[2]拉成1D的tensor, gbias.
      Assign(gbias, req[conv::kBias], sumall_except_dim<1>(grad));
      /*
	  Assign赋值操作, 赋值对象时gbias, 即损失关于卷积层的偏置bias的梯度; 数目草书模式是kBias的; exp是sumall_except_dim<1>(grad).
	  即除了第1维度的, 对上一层(第l + 1层)的残差进行求和. 
	  grad是4D的tensor. 上一层(第l + 1层)的损失关于输入的残差!  
	  
	  这个gbias和理论是可以对上的. 
	  */
    }
  }

 private:
  inline index_t InitTemp(const mshadow::Shape<4> &ishape,
                          const mshadow::Shape<4> &oshape) {
    const int ksize_y = param_.kernel[0];
    const int ksize_x = param_.kernel[1]; // 定义ksize_y和ksize_x, 分别是kernel[0]和kernel[1]. 
    shape_colunit_ = mshadow::Shape2(ishape[1] * ksize_y * ksize_x,
                                     oshape[2] * oshape[3]); // 利用ishape和oshape, 对shape_colunit_赋值. 定义temp_col. 
    shape_dstunit_ = mshadow::Shape3(param_.num_group,
                                     param_.num_filter / param_.num_group,
                                     oshape[2] * oshape[3]); // 利用oshape, 对shape_dstunit_赋值. 定义temp_dst. 
    // param_.workspace is in elements of sizeof(DType)
    // if param_.workspace is set to zero the nstep_ equals ishape[0] (batch)
    nstep_ = std::max(
        std::min(
            static_cast<index_t>(
                param_.workspace / (shape_colunit_.Size() + shape_dstunit_.Size())),
            ishape[0]),
        1U); // 对nstep_赋值. 做for循环. 
    // cout<<"nstep_: "<<nstep_<<endl; // batch_size. 

    mshadow::Shape<2> scol = mshadow::Shape2(shape_colunit_[0],
                                             shape_colunit_[1] * nstep_); // 定义一个2D的shape, 其大小是:
	// { shape_colunit_[0] } * { shape_colunit_[1] * nstep_ }. 
    mshadow::Shape<3> sdst = mshadow::Shape3(shape_dstunit_[0],
                                             shape_dstunit_[1],
                                             shape_dstunit_[2] * nstep_); // 定义一个3D的shape, 其大小为:
	// { shape_dstunit_[0] } * { shape_dstunit_[1] } * {shape_dstunit_[2] * nstep_} 
    index_t required_size = scol.Size() + sdst.Size();
    CHECK_GE(param_.workspace, required_size)
      << "\nMinimum workspace size: " << required_size * sizeof(DType) << " Bytes\n"
      << "Given: " << param_.workspace * sizeof(DType) << " Bytes";
    return required_size; // required_size就是 scol.Size() + sdst.Size(); 即上述shape2的大小(各个维数的大小相乘) + shape3的
	// 大小. 
  }

  ConvolutionParam param_;
  mshadow::Shape<2> shape_colunit_;
  mshadow::Shape<3> shape_dstunit_;
  index_t nstep_;
  /*
  shape_colunit_ , shape_dstunit_, nstep_是定义在类ConvolutionOp中的三个变量, 运行时输出一下. 
  shape_colunit_是Shape<2>定义的, 因此只有shape_colunit_[0]和shape_colunit_[1], 即是一个二维的shape;
  shape_dstunit_是Shape<3>定义的, 因此有shape_dstunit_[0], shape_dstunit_[1], shape_dstunit_[2].
  
  这三个量shape_colunit_, shape_dstunit_, nstep_. 
  是在运行时指定的, 是在ConvolutionOp类的私有函数InitTemp中指定的, 最新版本的convolution-inl.h中也有一个私有函数
  LayerSetUp, 用来指定一些值. 根据ishape(本层输入shape)和oshape(本层输出shape)可以确定OP中用到的一些变量的值. 
  */
};  // class ConvolutionOp

template<typename xpu>
Operator* CreateOp(ConvolutionParam param, int dtype,
                   std::vector<TShape> *in_shape,
                   std::vector<TShape> *out_shape,
                   Context ctx);

#if DMLC_USE_CXX11
class ConvolutionProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.kernel.ndim() == 2) {
      param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    } else {
      CHECK_EQ(param_.kernel.ndim(), 3) << param_.kernel.ndim() << "D convolution not supported";
      param_.layout = param_.layout ? param_.layout.value(): mshadow::kNCDHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape3(1, 1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
    }
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    // CHECK_EQ(out_shape->size(), 1) << "Output: [output]";
    out_shape->resize(1, TShape());
    const TShape &dshp = (*in_shape)[conv::kData];
    if (dshp.ndim() ==  0) return false;
    if (param_.kernel.ndim() == 2) {
      // 2d conv
      CHECK_EQ(dshp.ndim(), 4) \
          << "Input data should be 4D in batch-num_filter-y-x";
      Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);
      Shape<4> wshape = Shape4(param_.num_filter / param_.num_group, dshape[1] / param_.num_group,
                               param_.kernel[0], param_.kernel[1]);
      wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
      wshape[0] *= param_.num_group;
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kWeight, wshape);
      if (!param_.no_bias) {
        SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
      }

      const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
      const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
      CHECK_EQ(dshape[1] % param_.num_group, 0) \
          << "input num_filter must divide group size";
      CHECK_EQ(param_.num_filter % param_.num_group, 0) \
          << "output num_filter must divide group size";
      CHECK_GT(param_.kernel.Size(), 0) \
          << "incorrect kernel size: " << param_.kernel;
      CHECK_GT(param_.stride.Size(), 0) \
          << "incorrect stride size: " << param_.stride;
      CHECK_GT(param_.dilate.Size(), 0) \
          << "incorrect dilate size: " << param_.dilate;
      CHECK(ksize_y <= dshape[2] + 2 * param_.pad[0]
            && ksize_x <= dshape[3] + 2 * param_.pad[1])
          << "kernel size exceed input";
      Shape<4> oshape;
      oshape[0] = dshape[0];
      oshape[1] = param_.num_filter;
      oshape[2] = (dshape[2] + 2 * param_.pad[0] -
          (param_.dilate[0] * (ksize_y - 1) + 1)) / param_.stride[0] + 1;
      oshape[3] = (dshape[3] + 2 * param_.pad[1] -
          (param_.dilate[1] * (ksize_x - 1) + 1)) / param_.stride[1] + 1;
      SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
      return true;
    } else if (param_.kernel.ndim() == 3) {
      // 3d conv
      CHECK_EQ(dshp.ndim(), 5) \
        << "Input data should be 5D in batch-num_filter-depth-y-x";
      Shape<5> dshape = ConvertLayout(dshp.get<5>(), param_.layout.value(), kNCDHW);
      Shape<5> wshape = Shape5(param_.num_filter / param_.num_group, dshape[1] / param_.num_group,
                               param_.kernel[0], param_.kernel[1], param_.kernel[2]);
      wshape = ConvertLayout(wshape, kNCDHW, param_.layout.value());
      wshape[0] *= param_.num_group;
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kWeight, wshape);
      if (!param_.no_bias) {
        SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
      }

      const index_t ksize_d = static_cast<index_t>(param_.kernel[0]);
      const index_t ksize_y = static_cast<index_t>(param_.kernel[1]);
      const index_t ksize_x = static_cast<index_t>(param_.kernel[2]);
      CHECK_EQ(dshape[1] % param_.num_group, 0)
        << "input num_filter must divide group size";
      CHECK_EQ(param_.num_filter % param_.num_group, 0)
        << "output num_filter must divide group size";
      CHECK_GT(param_.kernel.Size(), 0) \
        << "incorrect kernel size: " << param_.kernel;
      CHECK_GT(param_.stride.Size(), 0) \
        << "incorrect stride size: " << param_.stride;
      CHECK_GT(param_.dilate.Size(), 0) \
        << "incorrect dilate size: " << param_.dilate;
      CHECK(ksize_d < dshape[2] + 2 * param_.pad[0]
            && ksize_y <= dshape[3] + 2 * param_.pad[1]
            && ksize_x <= dshape[4] + 2 * param_.pad[2])
        << "kernel size exceed input";
      CHECK_EQ(param_.dilate.Size(), 1)
        << "Dilate is not supported in 3d convolution";
      Shape<5> oshape;
      oshape[0] = dshape[0];
      oshape[1] = param_.num_filter;
      oshape[2] = (dshape[2] + 2 * param_.pad[0] -
          (1 * (ksize_d - 1) + 1)) / param_.stride[0] + 1;
      oshape[3] = (dshape[3] + 2 * param_.pad[1] -
          (1 * (ksize_y - 1) + 1)) / param_.stride[1] + 1;
      oshape[4] = (dshape[4] + 2 * param_.pad[2] -
          (1 * (ksize_x - 1) + 1)) / param_.stride[2] + 1;
      SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCDHW, param_.layout.value()));
      return true;
    } else {
      LOG(FATAL) << "Unknown convolution type";
      return false;
    }
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ConvolutionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Convolution";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[conv::kOut], in_data[conv::kData], in_data[conv::kWeight]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  ConvolutionParam param_;
};  // class ConvolutionProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONVOLUTION_INL_H_
