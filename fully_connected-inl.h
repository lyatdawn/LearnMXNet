/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_op-inl.h
 * \brief fully connect operator and symbol
*/
#ifndef MXNET_OPERATOR_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_FULLY_CONNECTED_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"


namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace fullc {
enum FullyConnectedOpInputs {kData, kWeight, kBias}; 
/*
全连接的输入包括:
上层输入数据: kData; 本层连接权重: kWeight; 本层连接偏置: kBias. 

在fully_connected-inl.h上先加上#include<iostream>, 然后再将kData, kOut, KBais输出, 再输出Shape的一些值. 与猜想一样, 在前向或
反向的过程中, kData, kOut, KBais是int型的数. 为0, 1, 2等数.  
*/ 
enum FullyConnectedOpOutputs {kOut}; // 输出: kOut. 
}  // fullc

struct FullyConnectedParam : public dmlc::Parameter<FullyConnectedParam> {
  int num_hidden;
  bool no_bias; 
  /*
  全连接层的参数: 
  num_hidden: 本层(全连接层)的结点个数.
  no_bias: 全连接层是否使用偏置. 
  */
  DMLC_DECLARE_PARAMETER(FullyConnectedParam) { // #define DMLC_DECLARE_PARAMETER(PType)
    // TODO(bing) add support for boolean
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1) // 全连接层的结点个数最少是1. set_lower_bound设置下界. 
    .describe("Number of hidden nodes of the output.");
    /*
	DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1) 利用宏DMLC_DECLARE_FIELD对全连接层的参数num_hidden进行描述, 并且参数有
	默认值或最小, 最大值.  
	*/
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    /*
	DMLC_DECLARE_FIELD(no_bias).set_lower_bound(1) 利用宏DMLC_DECLARE_FIELD对全连接层的参数no_bias进行描述, 并且参数有
	默认值或最小, 最大值. 默认使用偏置. 
	*/
  }
};

/**
 * \brief This is the implementation of fully connected operator.
 * \tparam xpu The device that the op will be executed on.
 */
 
/*
仅是全连接层, 无激活函数层:
a^(l) = W' * z^(l)
z^(l + 1) = a^(l)
a^(l + 1) = W' * z^(l + 1)
*/

template<typename xpu, typename DType> 
/*
全连接层的计算只有一种形式, 这里全连接层不包含激活函数, 即输出out = W .* X + b. 因此在定义FullyConnectedOp类时的模板参数只有
xpu和DType. 

在make mxnet的时候, 屏幕会根据config.mk给出xpu和DType的值. 
[with xpu = mshadow::op::cpu, DType = float] 
*/ 
class FullyConnectedOp : public Operator {
 public:
  explicit FullyConnectedOp(FullyConnectedParam p) {
    this->param_ = p;
    /*
    C++中的explicit关键字只能用于修饰只有一个参数的类构造函数, 它的作用是表明该构造函数是显示的, 而非隐式的, 跟
    它相对应的另一个关键字是implicit, 意思是隐藏的, 类构造函数默认情况下即声明为implicit(隐式).
    
    this指针是一个隐含于每一个成员函数中的特殊指针. 它指向正在被该成员函数操作的那个对象. 
    
    p是FullyConnectedParam全连接层参数类的对象, 将p赋值给param_. 这和单纯的赋值不一样, param_就是p, 可以看做是指向p的指针.
	*/
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    if (req[fullc::kOut] == kNullOp) return; // 全连接层的数据操作模式不能是kNullOp, 即什么都不做. 
    CHECK_EQ(req[fullc::kOut], kWriteTo); 
    /*
	case kNullOp:                     \
          break;                          \
        case kWriteTo:                    \
        case kWriteInplace:               \
          (out) = (exp);                  \
          break;                          \
        case kAddTo:                      \
          (out) += (exp); 
	*/
    
    size_t expected = param_.no_bias ? 2 : 3;
    /*
	param_.no_bias是否为真, 如果为真则expected为2, 否则为3. 
	*/
    
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    // TODO(bing): check the BLAS Handle, be careful
    // maybe need blas handle from context
    // TODO(bing): judge shape to remove flatten op
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__

    // std::cout<<"in_data kData: "<<fullc::kData<<std::endl; 这里fullc::kData是0, 即数据data是in_data[0]. 
    const TShape& ishape = in_data[fullc::kData].shape_;
    
    // std::cout<<"out_data kOut: "<<fullc::kOut<<std::endl; 这里fullc::kOut是0, 即数据data是out_data[0].
	/*
	在mxnet中, kData是0. 代表数据. 
	*/ 
    const TShape& oshape = out_data[fullc::kOut].shape_;
    /*
	定义输入数据in_data[0]的shape和输出数据out_data[0]的shape.
	
	TShape mxnet::TBlob::shape_, 可以用来定义Tensor的shape. TBlob类的成员, 返回值类型是TShape. 
	*/

	// std::cout<<"in_data kData: "<<fullc::kData<<std::endl; 这里fullc::kData是0, 即数据data是in_data[0].
	// std::cout<<"ishape: "<<ishape[0]<<" "<<ishape.ndim()<<std::endl; ishape[0]是64, 64是batch_size, 即批训练量的大小.  
	// ishape.ndim()是2, 即ishape是一个2维的. 
    Tensor<xpu, 2, DType> data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    /*
	将in_data[0](输入数据)拉成2维的张量. 这里将TBlob数据拉成Tensor数据时没有使用FlatTo2D, 而是用了get_with_shape. 定义如下:
	mshadow::Tensor<Device, dim, DType> mxnet::TBlob::get_with_shape(const mshadow::Shape<dim> & shape, 
	mshadow::Stream< Device > *stream = NULL)const
	给定shape, 将TBlob拉成一个Tensor. 如果shape和存储的大小不一致时, 会报错.
	
	在https://raw.githubusercontent.com/jdeng/gomxnet/master/mxnet.cc可以找到Shape1, Shape2, Shape3, Shape4的定义:
	Shape2定义如下:
	MSHADOW_XINLINE Shape<2> Shape2(index_t s0, index_t s1) {
        Shape<2> s; s[0] = s0; s[1] = s1;
        return s;
    } 因此, Shape2就是Shape<dim>类型的函数, Shape2返回Shape<2>的对象. 正好和get_with_shape函数的第一个参数相对应.
	
	Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())):
    index_t是一种索引类型(typedef mshadow::index_t mxnet::index_t), 其中s0是ishape[0]. s1是ishape.ProdShape(1, ishape.ndim()).
	
    index_t的定义在mshadow/mshadow/base.h下.
	typedef unsigned index_t; 
	unsigned a; 与unsigned int a; 是同样的. 这里省略了int, 只能和unsigned int等价.  
	
	ProdShape是类TShape类下的成员函数: 可以通过寻找TShape类来找到该函数:
	
	index_t mxnet::TShape::ProdShape(int dimstart, int dimend )const 
	生成一个索引, 索引属于[dimstart,dimend), 返回值类型是index_t, 即是一个索引类型. 因此, 
	ishape.ProdShape(1, ishape.ndim())就产生一个[1, 2)的索引.   
	*/    
    
	// std::cout<<"in_data kWeight: "<<fullc::kWeight<<std::endl; fullc::kWeight是1, 在in_data中kWeight代表1, 权重.    
    Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
    /*
	将本层(第l层)的权重in_data[kWeight]拉成2维的张量. 这次并没有使用get_with_shape, 而是使用get函数:
    mshadow::Tensor<Device, dim, DType> mxnet::TBlob::get(mshadow::Stream<Device> *stream = NULL)const 
	*/
    
    // std::cout<<"out_data kOut: "<<fullc::kOut<<std::endl; fullc::kOut是0, kData和kOut均是0.
	// std::cout<<"oshape: "<<oshape[0]<<" "<<oshape.ndim()<<std::endl; oshape和ishape一样, 64 2. 
    Tensor<xpu, 2, DType> out = out_data[fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);  
    /*
	将out_data[0](输出数据)拉成2维的张量. 这里将TBlob数据拉成Tensor数据时没有使用FlatTo2D, 而是用了get_with_shape. 定义如下:
	mshadow::Tensor<Device, dim, DType> mxnet::TBlob::get_with_shape(const mshadow::Shape<dim> & shape, 
	mshadow::Stream< Device > *stream = NULL)const
	给定shape, 将TBlob拉成一个Tensor. 如果shape和存储的大小不一致时, 会报错.
	
	定义out时和定义data的方法是一致的. 一个数本层(第l层)的输入, 一个是本层(第l层)的输出. 
	*/    
    
    out = dot(data, wmat.T());
    /*计算输出out. 
	根据全连接层的前向传播操作可知: 全连接层的输出 = W.*输入 + 偏置.
	
	因此如果全连接层没有偏置项, 那么 out = data .* weight. 
	
	dot函数: http://mxnet.io/api/python/symbol.html, 计算两个数组的点乘之积. 向量点乘(1D)或矩阵乘法(2D)... 
	
	wmat.T()是转置, 即权重的转置. 
	*/
    
    if (!param_.no_bias) { // 如果使用偏置, 那么out += ... + bias. 
      // std::cout<<"in_data kData: "<<fullc::kBias<<std::endl; fullc::kBias是2, 代表偏置. 
      Tensor<xpu, 1, DType> bias = in_data[fullc::kBias].get<xpu, 1, DType>(s);
      /*
	  将本层(第l层)的偏置in_data[kBias]拉成1维的张量. 这次并没有使用get_with_shape, 而是使用get函数:
      mshadow::Tensor<Device, dim, DType> mxnet::TBlob::get(mshadow::Stream<Device> *stream = NULL)const 
      
      1维的张量即是一个向量. 
	  */
      
      out += repmat(bias, data.size(0));
      /*这里也做一下输出, 看一下mxnet的全连接是如何实现的. 
      // std::cout<<"data.size: "<<data.size<<std::endl; Tensor张量的大小不能这样输出. 
	  // std::cout<<"bias: "<<bias<<std::endl; Tensor张量不能这样输出. 
	  
	  std::cout<<"data.size(0): "<<data.size(0)<<std::endl; 输出为64, 即batch_size. 
	  
	  =====================================================批处理过程========================================================
	  批处理的过程, 批处理大小是batch_size. 网络(上次更新完参数)->输入batch_size个样本进行正向传播, 
	  批处理时的正向传播是一次性将batch-size个样本输入, 得到batch_size个输出(简单地, 将batch_size看成1即可)
	  ->利用batch_size个标签进行反向传播, 更新网络的参数->得到更新完参数的网络->下一个batch的样本... 
	  ====================================================批处理过程=========================================================
	  
	  因此批处理时的全连接层data里有batch_size个样本的数据, out也是batch_size个输出, 但是网络就一个, 因此权重wmat.T()就一个.
	  然后全连接层如果有偏置项, 再加上偏置项即可. 
	  
	  out一共是batch_size个样本的输出, 因此没有个输出都应该加上一个bias, 而bias是一个列向量, 因此先将bias进行复制, 利用函数
	  repmat进行复制, 一共复制batch_size个, 然后再和 W.*X进行相加即可.
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
    /*全连接层(第l层)有权重和偏置, 因此要计算损失J关于权重的梯度和关于偏置的梯度. 也要计算残差.
	!!!!!!!!!!!!!!!!梯度可以看做是损失J关于层参数的导数, 残差可以看做是损失J关于层输入的导数!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
	 
    in_grad输出梯度参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的.
	out_grad输入残差参数, 向量容器, 每个元素的类型是TBlob. 上一层(第l + 1层)的残差, 计算本层的残差. 
	in_data输入参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的输入.  
	out_data输出参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的输出.	
	req: 数据操作模式, 向量数组. 元素类型是OpReqType.
	因为反向传播主要是计算梯度的, 因此in_data不参与计算. 
	*/
    
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    size_t expected = param_.no_bias ? 2 : 3;
    /*
	定义expected, 若全连接层的没有偏置, 则expected是2, 如果有偏置, expected是3. 即expected代表了in_data里TBolb对象的个数. 
	默认no_bias是false, 即有偏置项. 
	*/
	
	// CHECK宏加以断言标记, 来保证程序的严谨性.
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    // in_data[0], in_data[1], in_data[2]...
    CHECK_EQ(req.size(), expected);
    // 对于数据data, weight, bias有不同的数据操作模式.  
    
    // TODO(bing): check the BLAS Handle, be careful
    //  maybe need blas handle from context
    Stream<xpu> *s = ctx.get_stream<xpu>();
    
        // std::cout<<"in_data kData: "<<fullc::kData<<std::endl; 这里fullc::kData是0, 即数据data是in_data[0]. 
    const TShape& ishape = in_data[fullc::kData].shape_;
    
    // std::cout<<"out_data kOut: "<<fullc::kOut<<std::endl; 这里fullc::kOut是0, 即数据data是out_data[0].
	/*
	在mxnet中, kData是0. 代表数据. 
	*/ 
    const TShape& oshape = out_grad[fullc::kOut].shape_;
    /*
	定义输入数据in_data[0]的shape和输出残差out_grad[0]的shape.
	
	TShape mxnet::TBlob::shape_, 可以用来定义Tensor的shape. TBlob类的成员, 返回值类型是TShape. 
	*/
	
	// std::cout<<"in_data kData: "<<fullc::kData<<std::endl; 这里fullc::kData是0, 即数据data是in_data[0].
	// std::cout<<"ishape: "<<ishape[0]<<" "<<ishape.ndim()<<std::endl; ishape[0]是64, 64是batch_size, 即批训练量的大小.  
	// ishape.ndim()是2, 即ishape是一个2维的. 
    Tensor<xpu, 2, DType> data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    /*
	将in_data[0](本层(第l层)的输入数据)拉成2维的张量. 这里将TBlob数据拉成Tensor数据时没有使用FlatTo2D, 而是用了get_with_shape. 
	*/    
    
	// std::cout<<"in_data kWeight: "<<fullc::kWeight<<std::endl; fullc::kWeight是1, 在in_data中kWeight代表1, 权重.    
    Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
    /*
	将本层(第l层)的权重in_data[kWeight]拉成2维的张量. 这次并没有使用get_with_shape, 而是使用get函数.
	*/
    
    // std::cout<<"out_data kOut: "<<fullc::kOut<<std::endl; fullc::kOut是0, kData和kOut均是0.
	// std::cout<<"oshape: "<<oshape[0]<<" "<<oshape.ndim()<<std::endl; oshape和ishape一样, 64 2. 
    Tensor<xpu, 2, DType> grad = out_grad[fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);  
    /*
	将out_grad[0](残差)拉成2维的张量. 这里将TBlob数据拉成Tensor数据时没有使用FlatTo2D, 而是用了get_with_shape. 
	grad就代表上一层(第l + 1层的残差). 
	*/  

#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif

    //  backprop
    CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
    /*
	#define CHECK_NE(val1, val2) CHECK_OP(_NE, !=, val1, val2)
	权重梯度的数据操作模式不能是kWriteInplace. 一般情况下, 所有的out_data的数据操作类型应该是kWriteTo, 在计算表示gradient
	的tensor的时候, 我们最好是将梯度累加起来, req的类型应该是kAddTo, 表示应该调用+=操作.  
	*/
    
    // gradient of weight
    Tensor<xpu, 2, DType> gwmat = in_grad[fullc::kWeight].get<xpu, 2, DType>(s);
    Assign(gwmat, req[fullc::kWeight], dot(grad.T(), data));
    /*计算本层(第l层)权重的梯度, 这里仅是全连接层, 并没有激活函数作用.
	in_grad是本层(第l层)的梯度TBlob, 将in_grad[1](第l层关于权重的梯度)拉成2维的张量. 即gwmat代表第l层的损失 J 关于权重的梯度.
	权重是矩阵.
	 
	赋值操作, 数据操作模式是req[fullc::kWeight], 应该是kAddTo, 表示应该调用+=操作. 
	
	根据http://ufldl.stanford.edu/wiki/index.php/反向传导算法 计算损失 J 关于权重和偏置的最后结果, 损失关于第l层权重的梯度是:
	delta^(l + 1) * [a^(l)]'. 
	上一层(第l + 1层)的残差 * [本层(第l层)的输出数据]'. 因此失关于第l层损权重的梯度是: grad.T() 和 data 做点积. 矩阵乘法. 
	*/
    
    // gradient of bias
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> gbias = in_grad[fullc::kBias].get<xpu, 1, DType>(s);
      Assign(gbias, req[fullc::kBias], sum_rows(grad));
    }
    /*计算本层(第l层)偏置的梯度, 这里仅是全连接层, 并没有激活函数作用.
    in_grad是本层(第l层)的梯度TBlob, 将in_grad[2](第l层关于权重的偏置)拉成1维的张量. 即gbias代表第l层的损失 J 关于偏置的梯度.
    偏置是向量. 
    
	如果本层(第l层)全连接层使用偏置, 损关于第l层偏置的梯度是: delta^(l + 1).
	上一层(第l + 1层)的残差. 
	
	赋值操作, 数据操作模式是req[fullc::kWeight]. 损失关于本层(第l层)偏置的梯度是sum_rows(grad). 
	*/
    
    // gradient of data
    Tensor<xpu, 2, DType> gdata = in_grad[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    Assign(gdata, req[fullc::kData], dot(grad, wmat));
    /*计算本层(第l层)data的梯度. 即损失J关于FC层的残差, 这个残差并不会对下一次的FC层的前向传播产生影响, 但是会利用gdata计算FC
	层前一层(第l - 1)层的残差.
	 
	in_grad是本层(第l层)的梯度TBlob, 将in_grad[0](第l层关于权重的data)拉成2维的张量. 即gdata代表第l层的损失 J 关于data的梯度.
    data是矩阵. 
    值操作, 数据操作模式是req[fullc::kData]. 本层(第l层)损失关于data的梯度是: W' * delta^(l + 1). FC层没有激活函数! 
	本层的权重wmat和上一层(第l + 1层)的梯度做点乘. 这里应该是做矩阵乘法. 
	*/
  }

 private:
  FullyConnectedParam param_;
};  // class FullyConnectedOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(FullyConnectedParam param, int dtype,
                   std::vector<TShape> *in_shape,
                   std::vector<TShape> *out_shape,
                   Context ctx);

#if DMLC_USE_CXX11
class FullyConnectedProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

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
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    CHECK_EQ(out_shape->size(), 1);
    TShape dshape = (*in_shape)[fullc::kData];
    TShape oshape = (*out_shape)[0];
    // require data to be known
    if (dshape.ndim() ==  0) return false;

    index_t num_input = dshape.ProdShape(1, dshape.ndim());
    SHAPE_ASSIGN_CHECK(*in_shape, fullc::kWeight, Shape2(param_.num_hidden, num_input));
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, fullc::kBias, Shape1(param_.num_hidden));
    }

    SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape2(dshape[0], param_.num_hidden));
    if (oshape.ndim() != 0) {
      dshape[0] = oshape[0];
      SHAPE_ASSIGN_CHECK(*in_shape, fullc::kData, dshape);
    }
    return true;
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
    FullyConnectedProp* fc_sym = new FullyConnectedProp();
    fc_sym->param_ = this->param_;
    return fc_sym;
  }

  std::string TypeString() const override {
    return "FullyConnected";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[fullc::kOut], in_data[fullc::kData], in_data[fullc::kWeight]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[fullc::kData], in_grad[fullc::kData]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  FullyConnectedParam param_;
};  // class FullyConnectedSymbol
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_FULLY_CONNECTED_INL_H_
