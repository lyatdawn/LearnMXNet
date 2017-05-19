/*!
 * Copyright (c) 2015 by Contributors
 * \file dropout-inl.h
 * \brief
 * \author
*/

#ifndef MXNET_OPERATOR_DROPOUT_INL_H_
#define MXNET_OPERATOR_DROPOUT_INL_H_ // 定义宏 MXNET_OPERATOR_DROPOUT_INL_H_.
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "./operator_common.h" // src/operator下, mxnet的层一些常用的属性.
#include "./mshadow_op.h" // src/operator下, 定义了一些结构体. 这些结构体用来接收数据实现某些层的前向输出和反向输出, 如激活函数 
// 层有softplus, softplus_grad. 一个计算前向的输出, 一个计算反向的输出. 

#if defined(USE_STATIC_MKL) && defined(_OPENMP)
#include <omp.h> // OpenMP头文件.  
#include <sched.h>

#include <mkl_vml_functions.h>
#include <mkl_vsl.h> // MKL的一些头文件. 
#endif  // USE_MKL && _OPENMP // 是否使用MKL和OPENMP. 在make mxnet的时候, BLAS库使用的是OpenBLAS, 并不是MKL. 
// 即 USE_BLAS = openblas, 所以USE_STATIC_MKL = NONE; 而且, USE_NNPACK = 0; USE_MKL2017 = 0; USE_OPENMP = 1. 
// defined(USE_STATIC_MKL) && defined(_OPENMP)即: 是否定义了宏 USE_STATIC_MKL 和 _OPENMP. 

namespace dropout {
enum DropoutOpInputs {kData}; // Dropout层的输入, 只有数据kData. 
enum DropoutOpOutputs {kOut, kMask}; // Dropout层的输出有两个: 输出数据kOut和kMask. 0和1.  
enum DropoutOpForwardResource {kRandom}; // Dropout层前向传播资源, kRandom. 
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
}  // namespace dropout

namespace mxnet {
namespace op {

#if defined(USE_STATIC_MKL) && defined(_OPENMP)
static void bernoulli_generate(int n, double p, int* r) {
  int seed = 17 + rand_r() % 4096;
  int nthr = omp_get_max_threads();
# pragma omp parallel num_threads(nthr)
  {
    const int ithr = omp_get_thread_num();
    const int avg_amount = (n + nthr - 1) / nthr;
    const int my_offset = ithr * avg_amount;
    const int my_amount = std::min(my_offset + avg_amount, n) - my_offset;
    if (my_amount > 0) {
      VSLStreamStatePtr stream;
      vslNewStream(&stream, VSL_BRNG_MCG31, seed);
      vslSkipAheadStream(stream, my_offset);
      viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, my_amount,
        r + my_offset, p);
      vslDeleteStream(&stream);
    }
  }
}
#endif  // USE_MKL && _OPENMP // 是否使用MKL和OPENMP.

struct DropoutParam : public dmlc::Parameter<DropoutParam> { // Dropout层的参数设置和描述. 
  float p; // p是在训练过程中激活/抑制结点状态的概率值. 
  DMLC_DECLARE_PARAMETER(DropoutParam) {
    DMLC_DECLARE_FIELD(p).set_default(0.5) // p的默认值是0.5(set_default).  
    .set_range(0, 1) // set_range设置参数p的变化范围, 是0-1.  
    .describe("Fraction of the input that gets dropped out at training time"); // describe描述参数的用途, 字符串. 
  }
};  // struct DropoutParam

template<typename xpu, typename DType>
class DropoutOp : public Operator { // Dropout操作类DropoutOp. 模板类这有两个模板参数xpu(cpu or gpu)和DType(float). 
 public:
  explicit DropoutOp(DropoutParam param) {
    // C++中的explicit关键字只能用于修饰只有一个参数的类构造函数, 它的作用是表明该构造函数是显示的, 而非隐式的. param是参数类
	// 的对象, 利用param来访问Dropout的参数p. 
    this->pkeep_ = 1.0f - param.p; // pkeep_是real_t型的变量. real_t定义见dmlc-core/include/dmlc/data.h.
	// typedef float real_t; 
	// 另外data.h中还有index_t的定义: typedef unsigned index_t; the unsigned integer type 
	// pkeep_ = 1.0f - p. 
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    /*前向操作, 虚函数. 函数的实现在类中定义. 不需要返回值. 本层为第 l 层. 
	in_data: 本层输入data, 只有上层的输入.
	req: 数据操作模式. 
	out_data: 本层输出, out. 在训练的时候本层输出有两个.  
	*/
    using namespace mshadow;
    using namespace mshadow::expr;
    
    CHECK_EQ(in_data.size(), 1); // in_data容器大小为1, 即Dropout层的输入参数只有数据. 
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 2);
    }
    /*
	ctx是OpContext结构体定义的成员. OpContext结构体定义见include/mxnet/operator.h. 利用ctx成员访问结构变量is_train:
	int is_train; // operator是在进行 train 还是 test (is_train); 
	*/
    Stream<xpu> *s = ctx.get_stream<xpu>(); // operator在哪个device上运行
    
    Tensor<xpu, 2, DType> data = in_data[dropout::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[dropout::kOut].FlatTo2D<xpu, DType>(s);
    /*
	将in_data[dropout::kData]输入数据利用FlatTo2D拉成2维的张量data; 本层(第l层)的输入. 
	定义out_data[dropout::kOut]输出数据利用FlatTo2D拉成2维的张量out. 本层(第l层)的输出. 
	*/
    
    if (ctx.is_train) { // 网络在训练阶段. 
      Tensor<xpu, 2, DType> mask = out_data[dropout::kMask].FlatTo2D<xpu, DType>(s);
      /*
	  网络在训练阶段时, out_data容器的大小是2. 一个数本层的输出数据, 一个是kMask. 
	  义out_data[dropout::kMask]输出数据利用FlatTo2D拉成2维的张量mask. mask可以这样理解, 在Dropout层, 对输入结点多加了一道概率
	  流程:
	  	   
	  原来结点的输入值是 yi^(l), 加上概率之后变为 ri^(l) * yi^(l), ri^(l) ~ Bernoulli(p). 即Dropout层就可以看做是对网络的输入
	  数据data加上了一个概率值. 因为 ri^(l) ~ Bernoulli(p), 即0-1分布, 所以该结点可能激活可能抑制, 也因此减小了网络的规模, 但是
	  网络的实际参数数目是不变的. (改变的是输入的data, 并不是连接的参数. 以一定的概率抑制/激活该结点). 
	  
	  因此, mask扮演的就是 ri^(l) 的角色, 即网络第l层的每个结点的概率值, 得到了mask后, 再和本层(第l层)的输入数据data进行相乘即
	  可.   
	  */
      
#if defined(USE_STATIC_MKL) && defined(_OPENMP) // USE_MKL && _OPENMP // 使用MKL和OPENMP.
      DType* outptr = out.dptr_;
      DType* dataptr = data.dptr_;
      int* maskptr = reinterpret_cast<int*>(mask.dptr_);
      int count = mask.shape_[0]*mask.shape_[1];
      bernoulli_generate(count, this->pkeep_, maskptr);
  #pragma omp parallel for // OPENMP并行 
      for (int i = 0; i < count; ++i) {
        outptr[i] = dataptr[i] * maskptr[i];
      }
#else // 不使用MKL和OPENMP. 
      Random<xpu> *prnd = ctx.requested[dropout::kRandom].get_random<xpu, real_t>(s);
      /*
	  OpContext: 结构体, 定义在include/mxnet/operator.h中, 该结构体可以记录操作在前向和后向传播中的信息. ctx是结构体OpContext定
	  义的对象, requested是OPContext结构体下的函数:
      // brief Resources requested by the operator
  	  std::vector<Resource> requested; // 用来返回操作所需的资源. 
      ctx.requested返回的是一个向量容器, 我们需要的只是kRandom的资源配置, 即一个随机操作资源. 
	  ctx.requested[dropout::kRandom]就是一个Resource的对象. 再调用get_random函数.
	  
	  Resource结构体是mxnet操作所需资源结构体, 和NDArray类似. NDArray是一个多维的数组对象.
	  
	  get_random函数定义见: include/mxnet/resource.h下: get_random函数是定义在Resource结构体下的函数: 
      template<typename xpu, typename DType>
 	  inline mshadow::Random<xpu, DType>* get_random(mshadow::Stream<xpu> *stream) 
 	  get_random是随机数生成器. 
	  stream是device流; 返回一个随机数生成器, 类型是 mshadow::Random<xpu, DType>* . real_t即float, 即DType.
	  
	  利用ctx获取kRandom所需的资源对象, 在调用get_random得到一个随机数生成器, *prnd即是一个随机数生成器. *prnd是在device s下, 
	  real_t类型的随机数生成器.   
	  */
      
      mask = tcast<DType>(F<mshadow_op::threshold>(
             prnd->uniform(mask.shape_), pkeep_) * (1.0f / pkeep_));
      /*
	  mask扮演的就是 ri^(l) 的角色, 即网络第l层的每个结点的概率值. 现在来获取mask的值. mask是一个2维的张量, 即矩阵. 因为data是2
	  维的张量, 所以mask也是2维的张量.
	  
	  均匀采样, 采样概率是结点状态抑制的概率, 根据这个概率来抑制连接, 然后把mask全部除以pkeep_. 
      因此这里Dropout的概率值p是结点是抑制状态的概率值, 1-p即激活状态. 根据Dropout的理论知识, 在test/predict时, 每个weight要
	  激活状态的概率值, 即1-p. 这样把mask全部除以pkeep_, 在test/predict的时候就不需要乘以 1-p 了. 
	  
      首先来看F<mshadow_op::threshold>(prnd->uniform(mask.shape_), pkeep_):
	  mshadow用于表达式操作的类(DotExp, BinaryMapExp, UnaryMapExp):
	  BinaryMapExp(二叉图)是双目运算的表达式类, 在BinaryMapExp类下有F函数F< OP >(lhs, rhs)描述了一个双目运算;
	  DotExp是做点乘的类, 其中最常用的就是dot函数;
	  UnaryMapExp类是单目运算符的表达式类, 在UnaryMapExp类下有F函数.
	  这里, F<mshadow_op::threshold>(prnd->uniform(mask.shape_), pkeep_)是一个双目运算符. F< OP >(lhs, rhs)中的OP就是操作符,
	  即lhs和rhs做什么运算, 这里OP是mshadow_op::threshold, mshadow_op::threshold是定义在src/operator/mshadow_op.h下:
	  
	  threshold操作是mshadow_op.h下的结构体, threshold是用来获取Bernoulli mask的. 即threshold就是专门来做Dropout的.
	  threshold操作如下: 传入参数a和b, 返回 a < b ? DType(1.0f) : DType(0.0f). 
	  这里a是prnd->uniform(mask.shape_), b是pkeep_即结点抑制状态概率. 
	  
	  prnd->uniform(mask.shape_)是均匀采样, uniform函数定义见: mshadow/mshadow/random.h 143行. uniform在类
	  class Random<cpu, DType>下, 而prnd是Random类的对象, 所以可以引用uniform函数.   
	  template<int dim>
	  inline expr::ReshapeExp<Tensor<cpu, 1, DType>, DType, dim, 1> uniform(Shape<dim> shape). shape是Tensor的shape, 这里即
	  mask.shape_, shape_即代表一个Tensor的shape. dim是tensor的维数, 这里是2, 即张量的维数是2. uniform函数是[0, 1]的均匀分布, 
	  在[0, 1]间为1, 其余为0. 将 prnd->uniform(mask.shape_) 输出一下:
	  其类型是mshadow::expr::ReshapeExp<mshadow::Tensor<mshadow::cpu, 1, float>, float, 2, 1>.   
	  
	  prnd->uniform(mask.shape_)是一个1维的张量, 因此可以当做标量使用, 因此, a是prnd->uniform(mask.shape_), 即a可以是一个标量. 
	  ----------------------------------------------------------------------------------------------------------------------- 
	  tcast操作, 该函数定义见: mshadow/mshadow/expression.h 108行. 
	  template<typename DstDType, typename SrcDType, typename EType, int etype>
	  inline TypecastExp<DstDType, SrcDType, EType, (etype|type::kMapper)> tcast(const Exp<EType, SrcDType, etype> &exp){...}.
	  建立一个标量表达式. 
	  */       
             
      Assign(out, req[dropout::kOut], data * mask);
      /*
	  Assign赋值操作, out是本层(第l层)的输出, req是数据操作模式, exp是data * mask. exp即在数据上加了一道概率程序, 将结点的数据
	  值和概率值相乘. 概率服从伯努利分布, 即0-1分布. 
	  */
#endif  // USE_MKL && _OPENMP
    } else {
      Assign(out, req[dropout::kOut], F<mshadow_op::identity>(data));
      /*
	  如果不是训练阶段, 就不需要mask了, 因为在训练阶段生成mask的时候, 部除以pkeep_了, 因此在test/predict阶段, 网络的weight就不
	  需要再乘 1 - p了. 因此, exp就是data.
	  
	  F<mshadow_op::identity>(data)是一个单目运算符, 运算符是mshadow_op::identity, identity这个结构体实现的操作是输入DType a,
	  返回DType a. 即输入等于输出. 
	   
      将data赋值给本层(第l层)输出out. 
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
    /*Dropout层(第l层)没有权重和偏置, 因此要计算的是损失J关在Dropout层(第l层)的残差.
    !!!!!!!!!!!!!!!!梯度可以看做是损失J关于层参数的导数, 残差可以看做是损失J关于层输入的导数!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
	 
    in_grad输出残差参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的.
	out_grad输入残差参数, 向量容器, 每个元素的类型是TBlob. 上一层(第l + 1层)的残差, 计算本层的残差. 
	in_data输入参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的输入.  
	out_data输出参数, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的输出. 	
	req: 数据操作模式, 向量数组. 元素类型是OpReqType.
	*/
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    /*
	Dropout层(第l层)的out_grad, in_grad容器大小为1. 即只有输入的残差(第l + 1层)的残差, 输出残差(第l层的残差).  
	*/
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> grad = out_grad[dropout::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> mask = out_data[dropout::kMask].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> gdata = in_grad[dropout::kData].FlatTo2D<xpu, DType>(s);
    /*Dropout为第l层. 
	将第l + 1层的残差out_grad[0]利用FlatTo2D函数拉成2维的张量. 即残差和数据是一样的, 是2维的. grad.
	将第l层的输出out_data[1]利用FlatTo2D函数拉成2维的张量. mask. out_data容器大小为2, 即一个是本层的输出out_data[0], 一个是
	Dropout层的mask out_data[1]. 
	定义本层(第l层)的残差是2维的张量. gdata.  
	*/
    
#if defined(USE_STATIC_MKL) && defined(_OPENMP)
      DType* ingradptr = gdata.dptr_;
      DType* outgradptr = grad.dptr_;
      int* maskptr = reinterpret_cast<int*>(mask.dptr_);

      int count = mask.shape_[0]*mask.shape_[1];

  #pragma omp parallel for
      for (int i = 0; i < count; ++i) {
        ingradptr[i] = outgradptr[i] * maskptr[i];
      }
#else  // USE_MKL && _OPENMP 使用MKL和OPENMP. 本质和不使用MKL和OPENMP时的反向操作是一样的, 只是使用MKL和OPENMP时, 先用:
	   /*
	   DType* 定义float*的数组ingradptr(本层残差), outgradptr(上一层残差)和int*的数组maskptr(本层mask). 然后:
	   ingradptr[i] = outgradptr[i] * maskptr[i]; conut = mask.shape_[0]*mask.shape_[1];即mask矩阵的高度和宽度乘积.		    
	   */ 
      Assign(gdata, req[dropout::kData], grad * mask);
      /*
	  不使用MKL和OPENMP时, 本层(第l层)的残差gdata = grad * mask, 即上一层(第l + 1层)的残差 * 本层(第l层)的mask.  
	  */
#endif  // USE_MKL && _OPENMP 不使用MKL和OPENMP.  
  }

 private:
  real_t pkeep_;
};  // class DropoutOp


template<typename xpu>
Operator *CreateOp(DropoutParam param, int dtype);

#if DMLC_USE_CXX11
class DropoutProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 1);
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1);
    int dtype = in_type->at(0);

    if (dtype == -1) {
      LOG(FATAL) << "input type to dropout is not specified.";
      return false;
    }

    size_t nout = this->ListOutputs().size();
    out_type->clear();
    for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new DropoutProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Dropout";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[dropout::kOut], out_data[dropout::kMask]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[dropout::kOut], in_grad[dropout::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[dropout::kData], out_data[dropout::kOut]}};
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kRandom};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mask"};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  DropoutParam param_;
};  // class DropoutProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_DROPOUT_INL_H_

