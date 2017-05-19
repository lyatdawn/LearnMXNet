/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling-inl.h
 * \brief
 * \author
*/

#ifndef MXNET_OPERATOR_POOLING_INL_H_
#define MXNET_OPERATOR_POOLING_INL_H_
/*
#ifndef是"if not defined"的简写, 先测试要定义的宏变量是否被宏定义过. 定义mxnet的plloing操作宏. 
*/

#include <dmlc/logging.h> // mxnet的日志头文件. 在dmlc-core/include/dmlc下,.
#include <dmlc/parameter.h> // mxnet的参数头文件, 在dmlc-core/include/dmlc下, 定义参数的. 
#include <mxnet/operator.h> // 在include/mxnet下, 定义操作基类(operator), 操作属性类, 方法等. 对OP或Prop的函数进行声明.
#include <algorithm> // c++算法库. 
#include <map> // 关联式容器, 元素的值与某个特定的键相关联, 而并非通过元素在数组中的位置类获取.
#include <vector> // c++向量容器. 
#include <string> // c++字符串 
#include <utility> // utility头文件定义重载的关系运算符, 简化关系运算符的写入, 还定义了pair类型,
// pair类型是一种模板类型, 可以存储一对值.
#include "./operator_common.h" // src/operator下, mxnet的层一些常用的属性.

namespace mxnet {
namespace op {

namespace pool_enum { // 定义pooling操作的输出, 输出等时, 命名空间定义为pool_enum. 
enum PoolingOpInputs {kData}; // pooling操作的输入数据kData, 为0. 
enum PoolingOpOutputs {kOut}; // pooling操作的输出kOut, 为0. 
enum PoolingOpType {kMaxPooling, kAvgPooling, kSumPooling}; // pooling操作的类型, mxnet的pooling操作使用三种类型:
/*
kMaxPooling: 最大池化, 为0.
kAvgPooling: 平均池化, 为1.
kSumPooling: 求和池化, 为2. 
*/ 
enum PoolingOpPadConventionType {kValid, kFull}; // 因为pooling操作其实也是一种卷积操作, 因此设定卷积操作的类型.
/*卷积操作类型, 这和MATLAB的卷积函数conv和conv2是类似的, conv做向量间的卷积, 即一维卷积; conv2做二维卷积, 即矩阵卷积. 
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
操作时的A为      操作时B为 . 然后再对应元素相乘在相加即是卷积后的结果. 
0 0 0 0 0        4 3       
0 1 2 3 0        2 1 
0 4 5 6 0
0 7 8 9 0 
0 0 0 0 0

conv2(A, B, 'valid'), 返回卷积计算中没有补零部分的计算结果, 即是CNN中的卷积操作, 让卷积核对齐A即可. 不补零. 

conv2(A,B, 'same'), 返回和A同样大小的卷积后的结果, 利用full操作可以得到same操作的结果. 不在左上 
 
*/ 
}  // namespace pool_enum

/*
卷积输出的特征图的大小, 输出特征图尺寸 = [(输入特征图尺寸 + 2 * pad - 卷积核尺寸)/步长](向下取整) + 1. 
在卷积层的操作时, 还有膨胀卷积的时, 而pooling操作就是一个普通的普通的卷积. 没有膨胀系数. 

一般池化由于每一池化窗口都是不重复的, 所以stride = size(kernel). 另外还有重叠池化, 即相邻的池化窗口间是有重叠的.还有金字塔
池化SPP.  
*/

struct PoolingParam : public dmlc::Parameter<PoolingParam> { // pooling操作的参数类PoolingParam. 包括参数的描述以及初值等. 
  TShape kernel; // 池化窗口大小(卷积核), TShape: 一个shape类, 该类可以表示一个Tensor的形状. 利用TShape来表示Tensor的大小.  
  TShape stride; // 池化的移动步长.  
  TShape pad; // 原始数据的高度或宽度方向上补上0值的圈数. 
  int pool_type; // pooling的类型, 是int型的变量pool_type. 即0, 1, 2: kMaxPooling, kAvgPooling, kSumPooling. 
  int pooling_convention; // 做pooling操作的时候卷积的类型, 0, 1: kValid, kFull.
  bool global_pool; // bool类型的变量global_pool, 是否做全局池化. 
  DMLC_DECLARE_PARAMETER(PoolingParam) { // set_default函数设置参数的默认值; describe函数对参数进行描述. 
    DMLC_DECLARE_FIELD(global_pool).set_default(false) // global_pool默认值是false. 
    .describe("Ignore kernel size, do global pooling based on current input feature map. "
              "This is useful for input with different shape"); // 不管池化窗口的大小, 根据输入特征图做全局pooling, 这对输入
              // 特征图的大小不一样的情况下式有用的. global_pool先理解为全卷积.

    DMLC_DECLARE_FIELD(kernel)
    .enforce_nonzero()
    /*
    inline FieldEntry<mxnet::TShape> &enforce_nonzero() {
       this->enforce_nonzero_ = true;
       return this->self();
    }
    bool enforce_nonzero_;
    强制设为不为0. 
	*/
    .describe("pooling kernel size: (y, x) or (d, y, x)"); // pooling卷积核(池化窗口的大小). 卷积也分二维卷积核三维卷积, 和
	// convolution是类似的. 根据输入特征图的第三个维度来定. 

    DMLC_DECLARE_FIELD(pool_type)
    .add_enum("max", pool_enum::kMaxPooling)
    .add_enum("avg", pool_enum::kAvgPooling)
    .add_enum("sum", pool_enum::kSumPooling)
    .describe("Pooling type to be applied."); // 使用kMaxPooling等参数时, 用pool_enum命名空间限定其区域即可. 
    /*
	利用函数add_enum(const std::string &key, int value)来为pool_type添加参数. 添加一对键值key和value. 指定pooling的类型. 
	*/

    DMLC_DECLARE_FIELD(pooling_convention).set_default(pool_enum::kValid)
    .add_enum("full", pool_enum::kFull)
    .add_enum("valid", pool_enum::kValid)
    .describe("Pooling convention to be applied."
              "kValid is default setting of Mxnet and rounds down the output pooling size."
              "kFull is compatible with Caffe and rounds up the output pooling size.");
    /*
	利用函数add_enum(const std::string &key, int value)来为pooling_convention添加参数. 指定pooling操作中卷积的类型, 默认是
	pool_enum::kValid, 即将卷积核对齐特征图后, 在对应位置元素相乘再相加, 不补零. 
	*/

    int stride_shape[] = {1, 1}; // 定义一个一维数组stride_shape, 初值为[1, 1]. 然后再设置stride时将数组再转化为TShape类型. 
    DMLC_DECLARE_FIELD(stride).set_default(TShape(stride_shape, stride_shape + 2))
    .enforce_nonzero() // 强制不为0. 
    .describe("stride: for pooling (y, x) or (d, y, x)");
    /*
    pooling-inl.h中新版的mxnet对stride的初值进行了改变, 从原来的:
	int stride_shape[] = {1, 1}; 
	DMLC_DECLARE_FIELD(stride).set_default(TShape(stride_shape, stride_shape + 2))
	改为:
	DMLC_DECLARE_FIELD(stride).set_default(TShape())
    
    在c++中数组名是一个常量指针, 它指向数组的开头, 数组名加2表示把指针向下移两个单位. 如同一个指针加一个常数, 
	*(数组名+i)才等于数组名(i). 
	
	TShape定义: http://mxnet.io/doxygen/classmxnet_1_1TShape.html. TShape(stride_shape, stride_shape + 2)用法具体见TShape的
	构造函数:
    template<typename RandomAccessIterator >
    mxnet::TShape::TShape(RandomAccessIterator begin, RandomAccessIterator end)
	构造Tuple, Tuple是一个动态大小的数据结构, 可以存储少量元素类型相同的数据. RandomAccessIterator是一个typename.	 
    
	池化窗口(卷积核)移动步长, 设置默认值是TShape(stride_shape, stride_shape + 2), 即是一个Tuple. 对于TShape的数据, 可以正常
	输出.
	
	TShape stride = TShape(); 
	std::cout<<"stride: "<<stride<<std::endl; // stride是(), 即为空的. 
    int stride_shape[] = {1, 1};
    TShape a = TShape(stride_shape, stride_shape + 2);
    std::cout<<"a: "<<a<<std::endl; // a是(1, 1).  
	*/

    int pad_shape[] = {0, 0};
    DMLC_DECLARE_FIELD(pad).set_default(TShape(pad_shape, pad_shape + 2))
    .describe("pad for pooling: (y, x) or (d, y, x)"); // pad是补零的圈数, 也有二维和三维的区别, 看输入特征图是二维的还是三
	// 维的. 
    /*
	pooling-inl.h中新版的mxnet对pad的初值进行了改变, 从原来的:
	int pad_shape[] = {1, 1}; 
	DMLC_DECLARE_FIELD(stride).set_default(TShape(pad_shape, pad_shape + 2))
	改为:
	DMLC_DECLARE_FIELD(pad).set_default(TShape())
	*/
  }
};

/*
* [pool](#pool): do pooling on image
* [unpool](#unpool): get gradient of pooling result
* [crop](#crop): crop the original image to a smaller size

使用sublime的全局查找功能, 来寻找pool函数和unpool函数的定义. 在本机上用sublime打开pooling-inl.h, 然后再使用sublime的全局查找
功能即可查询到函数的定义. 

以后寻找函数的定义, 可以参考 mxnet 官方文档以及使用sublime的全局查找功能实现. 
*/

template<typename xpu, typename Reducer, typename DType> // 模板类定义时, 一般都有xpu(cpu or gpu)和Dtype(float). pooling操作
// 这加了一个Reducer, 池化类型. 
class PoolingOp : public Operator {
 public:
  explicit PoolingOp(PoolingParam p) {
    this->param_ = p; // explicit关键字只能用于修饰只有一个参数的类构造函数, 它的作用是表明该构造函数是显示的, 而非隐式的.
    // p是PoolingParam卷积层参数类的对象, 将p赋值给param_. 这和单纯的赋值不一样, param_就是p, 可以看做是指向p的指针.
    // 利用param_来访问PoolingParam类的成员, 如kerne, stride等参数. 
    // PoolingParam param_;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    /*前向操作, 虚函数. 函数的实现在类中定义. 不需要返回值. 本层为第 l 层. 
	in_data: 本层输入data. pooling操作层是没有权重和偏置的, 根据以往的池化层的操作, max池化就是取池化窗口对应的特征图中的元
	素最大值; avg就是取元素的平均值; sum就是求和. 所以并不需要权重wmat和偏置bias. 
	req: 数据操作模式. 
	out_data: 本层输出, out. 
	*/
    using namespace mshadow;
    using namespace mshadow::expr;
    
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    /*
	判断输入和输出的TBlob个数, 容器大小应该是1. 即pooling操作的in_data只有kData, out_data只有kOut. 所以容器大小为1, 若不为1,
	断言.  
	*/
    
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.kernel.ndim() == 3) {
      LOG(FATAL) << "3D kernel not implemented";
    } // 3维的卷积也是没有实现的. 就像caffe的数据结构blob一样, blob是四维的结构, 单个数据是二维的, 因此做不了三维的. 
    
    Tensor<xpu, 4, DType> data = in_data[pool_enum::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[pool_enum::kOut].get<xpu, 4, DType>(s);
    /*引用kData个kOut需要指定命名空间, 指定其定义域. 
	将本层(第l层)的输入数据in_data[kData], out_data[pool_enum::kOut拉成4维的张量. 这使用get函数:
    mshadow::Tensor<Device, dim, DType> mxnet::TBlob::get(mshadow::Stream<Device> *stream = NULL)const. 4维的张量, 这里就和
	blob比较类似了, 即number N x channel K x height H x width W, (N, K, H, W). Blob memory is row-major in layout. 
	*/
    
    mshadow::Shape<2> out_shape = Shape2(out.shape_[2], out.shape_[3]);
    /*
	定义一个2为的shape out_shape, Shape2定义如下:
	MSHADOW_XINLINE Shape<2> Shape2(index_t s0, index_t s1) {
        Shape<2> s;
        s[0] = s0; s[1] = s1;
        return s;
    } 
    这里对out_shape赋值, 采用的是out.shape_[2], out.shape_[3], 即本层(第l层)输入数据的高度H和宽度W. out_shape即本层输出数据的
	大小. 
    out.shape_[0]: number N 
    out.shape_[1]: channel K 
	out.shape_[2]: height H 
	out.shape_[3]: width W 
	*/
    
    if (param_.pool_type == pool_enum::kMaxPooling || param_.pool_type == pool_enum::kSumPooling) { // param_.pool_type访问
	// 池化的类型. 最大池化和求和池化: 
      Assign(out,
             req[pool_enum::kOut],
             pool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                           out_shape,
                           param_.global_pool ? data.shape_[2] : param_.kernel[0],
                           param_.global_pool ? data.shape_[3] : param_.kernel[1],
                           param_.global_pool ? 1 : param_.stride[0],
                           param_.global_pool ? 1 : param_.stride[1]));
    /*
	赋值操作. pooling输入是data, 输出是out, 给out赋值.
	Assign操作定义在include/mxnet/operator_util.h下, 是定义的一个宏函数. 根据需求将exp的值赋给out. 
	这不是C++的字符串赋值函数assign. 
	#define ASSIGN_DISPATCH(out, req, exp)  \
  	{                                     \
      switch (req) {                      \
        case kNullOp:                     \
          break;                          \
        case kWriteTo:                    \
        case kWriteInplace:               \
          (out) = (exp);                  \
          break;                          \
        case kAddTo:                      \
          (out) += (exp);                 \
          break;                          \
        default:                          \
          LOG(FATAL) << "not reached";    \
      }                                   \
    } 
	
	pool_enum::kOut就是获取一种枚举类型(找到OpReqType类型的那个索引), 那么req[pool_enum::kOut]即OpReqType中的kWriteInplace或
	kAddTo, 然后通过exp给out赋值.
	
	Assign的赋值操作中exp(计算结果)为 pool<Reducer>(...). 在mshadow/doc/README.md下有 pool 和 unpool 的一些介绍:
    pool<Reducer>(Expr<xpu, dim> img, [Shape<2> pshape,] int ksize_y, int ksize_x, int kstride) 给定池化窗口的大小和滑动步长
	做池化操作. Reducer是一个输入参数, operation can be max or sum. 
	
	pool函数的具体定义在mshadow/mshadow/extension/spatial_pool.h下, unpool定义在mshadow/mshadow/extension/spatial_unpool.h下.
	241行使用的pool函数, 其第二个参数是out_shape, 类型是Shape<2>, 因此241行使用的pool函数定义如下: 定义在 mshadow::expr 命名
	空间下, 因此要using namespace mshadow::expr; 
	
	template<typename Reducer, typename SrcExp, typename DType, int etype>
	pool(const Exp<SrcExp, DType, etype> &src, Shape<2> pshape,
     index_t ksize_y, index_t ksize_x, index_t kstride_y, index_t kstride_x)
	src是源图像; pshape是经pooling后输出特征图的shape, 类型是Shape<2>; ksize_y核在y方向的大小(高); ksize_x核宽度; 
	kstride_y步长高度; kstride_x步长宽度. 返回池化后的结果. 但是241行使用pool操作的时候, src是利用pad函数定义的:
	
	pad也是定义下mshadow::expr命名空间下, 因此要using namespace mshadow::expr; 定义见mshadow/mshadow/extension/pad.h. 由于241
	行使用的pad有三个参数, 因此:
	template<typename SrcExp, typename DType, int etype>
	pad(const Exp<SrcExp, DType, etype> &src, index_t pad_y, index_t pad_x)
	对一张图片进行补零操作, 在图片的四周补零. src原图像; pad_y padding size in y, 即在y方向上补零的行数; pad_x 
	padding size in x, 在x方向上补零的列数. 返回补零的结果, 即返回补完零之后的矩阵. 
	
	pad(data, param_.pad[0], param_.pad[1])即对Tensor<xpu, 4, DType>的data进行补零操作. 返回的是补完零后的矩阵.
	out_shape是输出特征图的高度和宽度, 类型是Shape<2>.
	param_.global_pool ? data.shape_[2] : param_.kernel[0], 是否做全局pool(global_pool是否为真, global_pool默认为假), 因此
	ksize_y是param_.kernel[0], 即核的高度; global_pool 为真, ksize_y为data.shape_[2], 即是样本矩阵的高度.  
    param_.global_pool ? data.shape_[3] : param_.kernel[1], global_pool默认为假, 因此ksize_x为param_.kernel[1], 即和的宽度.
	global_pool 为真, ksize_x为data.shape_[3], 即是样本矩阵的宽度. 
    param_.global_pool ? 1 : param_.stride[0], global_pool默认为假, 因此kstride_y为param_.stride[0], 滑动步长的高度; 
	global_pool 为真, 此kstride_y为1. 
    param_.global_pool ? 1 : param_.stride[1], global_pool默认为假, 因此kstride_x为param_.stride[1], 滑动步长的宽度; 
	global_pool 为真, 此kstride_x为1.   
	
	池化的类型. 最大池化和求和池化: 
	*/
    
    } else if (param_.pool_type == pool_enum::kAvgPooling) { // 池化类型: 平均池化. 
      Assign(out,
             req[pool_enum::kOut],
             scalar<DType>(1.0f / (param_.global_pool ?
                      data.shape_[2] * data.shape_[3] :
                      param_.kernel[0] * param_.kernel[1])) *  // C++中 \ 是为了书写方便时使用的, 换行. 有 \ 会出错. 
             pool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                           out_shape,
                           param_.global_pool ? data.shape_[2] : param_.kernel[0],
                           param_.global_pool ? data.shape_[3] : param_.kernel[1],
                           param_.global_pool ? 1 : param_.stride[0],
                           param_.global_pool ? 1 : param_.stride[1]));
    /*
	Assign赋值操作, 将exp的值通过数据操作模式req传递给输出out. 做平均池化和最大池化, 求和池化的exp不同, 平均池化对pool(....)
	的结果利用scalar进行伸缩了. 
	
	scalar<DType>见mshadow/mshadow/expression.h:
	template<typename DType>
	inline ScalarExp<DType> scalar(DType s) 建立一个标量(scalar)表达式. DType是data的类型, 如float; s是一个value. 这里s是一
	个表达式:
    1.0f / (param_.global_pool ? data.shape_[2] * data.shape_[3] : param_.kernel[0] * param_.kernel[1])
	global_pool默认为假, 因此(..)为param_.kernel[0] * param_.kernel[1], 即核的尺寸乘积; global_pool 为真, (..)为
	data.shape_[2] * data.shape_[3], 即样本矩阵的尺寸. 然后再执行 1.0f / (...) 就是s值.   
	
	Reducer是一个输入参数, 这里是avg. 
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
    /*池化层(第l层)没有权重和偏置, 因此要计算的是损失J关在池化层(第l层)的残差.
    !!!!!!!!!!!!!!!!梯度可以看做是损失J关于层参数的导数, 残差可以看做是损失J关于层输入的导数!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
	 
    in_grad输出残差参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的.
	out_grad输入残差参数, 向量容器, 每个元素的类型是TBlob. 上一层(第l + 1层)的残差, 计算本层的残差. 
	in_data输入参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的输入.  
	out_data输出参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的输出. 	
	req: 数据操作模式, 向量数组. 元素类型是OpReqType.
	*/
							  
    using namespace mshadow;
    using namespace mshadow::expr;
    
    CHECK_EQ(out_grad.size(), 1); // 上一层(第l + 1层)反向传播中只有损失的残差sigma^(l + 1). 若不然, 断言. 
    CHECK_EQ(in_data.size(), 1); // 本层(第l层)的输入只有数据. 
    CHECK_EQ(out_data.size(), 1); // 本层(第l + 1层)的输出只有数据. 
    CHECK_EQ(req.size(), 1); // 数据操作模式只有一种.  
    CHECK_EQ(in_grad.size(), 1); // 本层(第l层)反向传播中只有残差. 
    
    // TODO(bing): remove pad (0,0)
    if (param_.kernel.ndim() == 3) {
      LOG(FATAL) << "3D kernel not implemented";
    } // 核的维数只能是2维的. 
    Stream<xpu> *s = ctx.get_stream<xpu>();
    
    // pool_enum::kOut为0; pool_enum::kData为0. 
    Tensor<xpu, 4, DType> grad = out_grad[pool_enum::kOut].get<xpu, 4, DType>(s);
    // 将第l + 1层的残差out_grad[0]利用get函数拉成四维的张量. 即残差和数据是一样的, 是4维的. 
    Tensor<xpu, 4, DType> data = in_data[pool_enum::kData].get<xpu, 4, DType>(s);
    // 将第l层的输入in_data[0]利用get函数拉成4维的张量. 
    Tensor<xpu, 4, DType> output_data = out_data[pool_enum::kOut].get<xpu, 4, DType>(s);
    // 将第l层的输出利用get函数拉成4维的张量. 
    Tensor<xpu, 4, DType> input_grad = in_grad[pool_enum::kData].get<xpu, 4, DType>(s);
    // 定义本层(第l层)的残差是4维的张量. 

    mshadow::Shape<2> in_shape = Shape2(data.shape_[2], data.shape_[3]); // 定义in_shape, 类型是Shape<2>. 是第l层的输入的样本
	// 矩阵大大小. data.shape_[2]为高度, data.shape_[1]为宽度. 反向传播时, 要将本层的残差reshape成和本层的输入一样大小. 即:
	// 本层的 残差 和 本层的输入 的shape是一致的. 

    if (param_.pool_type == pool_enum::kMaxPooling || param_.pool_type == pool_enum::kSumPooling) { // 最大池化和求和池化. 
      Assign(input_grad, req[pool_enum::kData],
             crop(unpool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                                  pad(output_data, 0, 0),
                                  pad(grad, 0, 0),
                                  param_.global_pool ? in_shape[0] : param_.kernel[0],
                                  param_.global_pool ? in_shape[1] : param_.kernel[1],
                                  param_.global_pool ? 1 : param_.stride[0],
                                  param_.global_pool ? 1 : param_.stride[1]),
                  in_shape,
                  param_.pad[0],
                  param_.pad[1]));
    /*
	Assign, 赋值操作. 通过req数据操作模式(req[0]), 将exp的值赋给input_grad, 即本层的残差. 其中exp是:
    crop(*, in_shape, param_.pad[0], param_.pad[1]). * 是unpool<Reducer>(...).
	
	crop函数定义见: mshadow/mshadow/extension/crop.h. 定义在mshadow::expr命名空间下, 因此要using namespace mshadow::expr;
	crop函数也有两种定义, 由于383行的crop有四个输入参数, 因此crop定义如下:
    template<typename SrcExp, typename DType, int etype>
	crop(const Exp<SrcExp, DType, etype> &src, Shape<2> oshape, index_t start_height, index_t start_width){....} 
	src是源图像; 
	oshape是裁剪后的数据的大小, in_shape; 
	start_height裁剪起始高度值; 
	start_width裁剪起始宽度值; 
	返回裁剪后的数据
	383行使用 crop(*, in_shape, param_.pad[0], param_.pad[1]). * 是对上一层的残差进行上采样得到的结果; in_shape即oshape, 是本
	输入的大小; 裁剪的起始位置是(param_.pad[0], param_.pad[1]). 
	
	* 是对上一层的残差进行上采样得到的结果. 是通过使用 unpool 函数对上一层的残差进行上采样得到, 具体过程如下:
	1> 最大池化:
	由于一般情况下是一般池化, 即 stride = kerne. 例如, kernel = (2, 2). 在反向传播时, 只有那个最大值对下一层有贡献,
	所以将残差传递到该最大值的位置, 区域内其他2*2-1=3个位置置零. 即
	 
	1       2       反向传播        0 0 0 0
	               ---------->      0 1 2 0            
	3       4                       0 3 4 0
	  残差                          0 0 0 0
	
	2>平均池化:
    如, kernel = (2, 2). 我们需要把残差平均分成2*2=4份, 传递到前边小区域的4个单元即可. 即:
	
	1       2       反向传播        1/4 1/4 1/2 1/2 
	               ---------->      1/4 1/4 1/2 1/2            
	3       4                       1/3 1/3 1   1  
	  残差                          1/3 1/3 1   1 
 
    unpool函数定义如下:
	template<typename Reducer, typename SrcExp, typename DType, int etype>
	unpool(const Exp<SrcExp, DType, etype> &data_src,
       const Exp<SrcExp, DType, etype> &data_pooled,
       const Exp<SrcExp, DType, etype> &grad_pooled,
       index_t ksize_y, index_t ksize_x, index_t kstride_y, index_t kstride_x). 对四维数据进行上采样, 获得池化层的参禅.
	data_src为本层(第l层)的输入, 即池化层的输入, 利用pad(...)函数补零的数据;
    data_pooled是本层(第l层)的输出, 即池化操作后的特征图, 利用pad(...)函数补零的数据; 
	grad_pooled为上一层(第l + 1层)的残差, 利用pad(...)函数补零的数据;
	ksize_y核的高度, global_pool默认为假, 因此ksize_y是param_.kernel[0], 即核的高度; global_pool 为真, ksize_y为data.shape_[2],
    即是样本矩阵的高度; 
    ksize_x核的宽度, global_pool默认为假, 因此ksize_x为param_.kernel[1], 即和的宽度. global_pool 为真, ksize_x为data.shape_[3],
    即是样本矩阵的宽度. 
    kstride_y为y方向上的滑动步长, global_pool默认为假, 因此kstride_y为param_.stride[0], 滑动步长的高度; global_pool 为真, 
	此kstride_y为1. 
    kstride_x为x方向上的滑动步长, global_pool默认为假, 因此kstride_x为param_.stride[1], 滑动步长的宽度; global_pool 为真, 
	此kstride_x为1.  
	*/
                  
    } else if (param_.pool_type == pool_enum::kAvgPooling) { // 平均池化. 
      Assign(input_grad, req[pool_enum::kData],
             scalar<DType>(1.0f / (param_.global_pool ?
                      data.shape_[2] * data.shape_[3] :
                      param_.kernel[0] * param_.kernel[1])) * \
             crop(unpool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                                  pad(output_data, 0, 0),
                                  pad(grad, 0, 0),
                                  param_.global_pool ? in_shape[0] : param_.kernel[0],
                                  param_.global_pool ? in_shape[1] : param_.kernel[1],
                                  param_.global_pool ? 1 : param_.stride[0],
                                  param_.global_pool ? 1 : param_.stride[1]),
                  in_shape,
                  param_.pad[0],
                  param_.pad[1]));
    /*
	这和前向的处理是一样的, 将crop(...)的结果伸缩, 乘上一个标量表达式scalar<DType>(DType s).
	
	1.0f / (param_.global_pool ? data.shape_[2] * data.shape_[3] : param_.kernel[0] * param_.kernel[1])
	global_pool默认为假, 因此(..)为param_.kernel[0] * param_.kernel[1], 即核的尺寸乘积; global_pool 为真, (..)为
	data.shape_[2] * data.shape_[3], 即样本矩阵的尺寸. 然后再执行 1.0f / (...) 就是s值. 
	*/  
    }
  }

 private:
  PoolingParam param_;
};  // class PoolingOp

template<typename xpu>
Operator* CreateOp(PoolingParam param, int dtype);


#if DMLC_USE_CXX11
class PoolingProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 1);
    const TShape &dshape = (*in_shape)[0];
    CHECK_GE(dshape.ndim(), 4) << "Pooling: Input data should be 4D in (batch, channel, y, x) "
                               << "Or 5D in (batch, channel, d, y, x)";
    TShape oshape = dshape;
    if (dshape.ndim() ==  0) return false;
    if (param_.kernel.ndim() == 2) {
      CHECK_EQ(dshape.ndim(), 4) << "Pooling: Input data should be 4D in (batch, channel, y, x)";
      if (param_.global_pool) {
        oshape[2] = 1;
        oshape[3] = 1;
      } else {
        CHECK(param_.kernel[0] <= dshape[2] + 2 * param_.pad[0])
            << "kernel size (" << param_.kernel[0] << ") exceeds input (" << dshape[2]
            << " padded to " << (dshape[2] + 2*param_.pad[0]) << ")";
        CHECK(param_.kernel[1] <= dshape[3] + 2 * param_.pad[1])
            << "kernel size (" << param_.kernel[1] << ") exceeds input (" << dshape[3]
            << " padded to " << (dshape[3] + 2*param_.pad[1]) << ")";
        if (param_.pooling_convention == pool_enum::kValid) {
          oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
                              param_.stride[0];
          oshape[3] = 1 + (dshape[3] + 2 * param_.pad[1] - param_.kernel[1]) /
                              param_.stride[1];
        } else {
          oshape[2] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[2] + 2 * param_.pad[0] -
                              param_.kernel[0]) / param_.stride[0]));
          oshape[3] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[3] + 2 * param_.pad[1] -
                              param_.kernel[1]) / param_.stride[1]));
        }
      }
      out_shape->clear();
      out_shape->push_back(oshape);
    } else if (param_.kernel.ndim() == 3) {
      CHECK_EQ(dshape.ndim(), 5) << "Pooling: Input data should be 5D in (batch, channel, d, y, x)";
      CHECK_LT(param_.kernel[0], dshape[2] + 2 * param_.pad[0]) << "kernel size exceeds input";
      CHECK_LE(param_.kernel[1], dshape[3] + 2 * param_.pad[1]) << "kernel size exceeds input";
      CHECK_LE(param_.kernel[2], dshape[4] + 2 * param_.pad[2]) << "kernel size exceeds input";
      if (param_.global_pool) {
        oshape[2] = 1;
        oshape[3] = 1;
        oshape[4] = 1;
      } else {
        if (param_.pool_type == pool_enum::kValid) {
          oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
                              param_.stride[0];
          oshape[3] = 1 + (dshape[3] + 2 * param_.pad[1] - param_.kernel[1]) /
                              param_.stride[1];
          oshape[4] = 1 + (dshape[4] + 2 * param_.pad[2] - param_.kernel[2]) /
                              param_.stride[2];
        } else {
          oshape[2] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[2] + 2 * param_.pad[0] -
                              param_.kernel[0]) / param_.stride[0]));
          oshape[3] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[3] + 2 * param_.pad[1] -
                              param_.kernel[1]) / param_.stride[1]));
          oshape[4] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[4] + 2 * param_.pad[2] -
                              param_.kernel[2]) / param_.stride[2]));
        }
      }

      out_shape->clear();
      out_shape->push_back(oshape);
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1);
    int dtype = (*in_type)[0];

    if (dtype == -1) {
      LOG(FATAL) << "Input type to pooling is not specified.";
      return false;
    }

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    PoolingProp *prop_sym = new PoolingProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "Pooling";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[pool_enum::kOut], in_data[pool_enum::kData], out_data[pool_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
#if MXNET_USE_CUDNN == 1
    return {};
#else
    return {{in_data[pool_enum::kData], in_grad[pool_enum::kData]}};
#endif
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  PoolingParam param_;
};  // class PoolingProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_POOLING_INL_H_
