/*!
 * Copyright (c) 2017 by Contributors
 * \file sofeplus-inl.h
 * \brief Sofeplus operator
 * \author
 * \ 本例依照activation-inl.h编写, 将Softplus激活函数做成一个op, 目的是熟悉mxnet利用C添加层的操作. 
 * \ 激活函数只有一种. 现在只涉及到operator这一块, 就需要将每一个op中包含的.h文件浏览一遍, 这样就对mxnet的op会有重新的认识了. 
 
 * \ 1>http://mxnet.io/doxygen/是mxnet的官方文档, 里面记载了mxnet的类, 方法的具体定义和实现. 该网站和dlib的文档网站一样. 
 * \ Doxygen是一种开源跨平台的，以类似JavaDoc风格描述的文档系统.
 * \ 2>Operators in MXNet网页, 对mxnet的op进行了简单地介绍, 可以参考. 
 * \ 3>http://mxnet.io/api/python/index.html#table-of-contents 是python的API. 
     4>Sublime的全局搜索功能, 可以在一个库/包中, 搜索关键字. 得到类, 函数, 变量等的定义. 最好用! 
	 5>mxnet将一个类, 结构体, 函数定义是写在了一个宏里. 
 * \
*/
#ifndef MXNET_OPERATOR_SOFTPLUS_INL_H_ // if not defined. 
#define MXNET_OPERATOR_SOFTPLUS_INL_H_

#include <dmlc/logging.h> // mxnet的日志头文件. 在dmlc-core/include/dmlc/下, 
#include <dmlc/parameter.h> // mxnet的参数头文件, 在dmlc-core/include/dmlc/parameter.h下, 定义一些参数的. 
#include <mxnet/operator.h> // 在include/mxnet下, 定义操作基类(operator), 操作属性类, 方法等. 对OP或OpProp的函数进行声明.  
// 但不实现, 然后mxnet在定义每一个OP时, 会重写这些函数, 以实现某一个OP的功能. 
#include <cstring> // c字符串. 
#include <map> // 关联式容器, 元素的值与某个特定的键相关联, 而并非通过元素在数组中的位置类获取. 如:
/*
std:map<int, string> personnel;
这样就定义了一个以int为键, 值为string的容器mappersonnel.
*/ 
#include <string> // c++字符串 
#include <vector> // 向量容器-数组. 
#include <utility> // utility头文件定义重载的关系运算符, 简化关系运算符的写入, 还定义了pair类型,
// pair类型是一种模板类型, 可以存储一对值.
#include "./operator_common.h" // src/operator下, mxnet的层一些常用的属性.

namespace mxnet { // mxnet命名空间. 
namespace op { // op(操作, 层)命名空间. 

namespace softplus { // 每个层(操作)都可以看做是一个命名空间. 
enum SoftplusOpInputs {kData}; // Softplus的输入-kData. // 枚举类型: enum 枚举类型名称 {变量值列表/枚举元素列表}; //
enum SoftplusOpOutputs {kOut}; // Softplus的输出-kOut. 
enum SoftplusOpType {kSoftplus}; // Softplus操作的类型, 这里激活函数就是Softplus.
}  // activation 
/*
枚举类型的变量值也可以看成是索引. 用来表示需要的变量. 

在softplus-inl.h上先加上#include<iostream>, 然后再将kData, kOut, KBais输出, 再输出Shape的一些值. 在前向或
反向的过程中, kData, kOut, KBais是int型的数. 为0, 0, 2等数. 

这是枚举类型的本身定义确定的. enum SoftplusOpInputs {kData}; 
SoftplusOpInputs是枚举类型名, 这个类型名其实没有什么实际的意义, 就是为了说明枚举便令kData的含义来的. SoftplusOpInputs并不是
类型或者类等.  
另外, 枚举元素按照常量处理, 即在以后的使用中不能改变枚举元素的值; 在定义的时候具有默认值, 从0开始递增, 0, 1, 2, ...; 
在定义枚举元素的时候可以对枚举元素进行赋初值操作. 如 enum SoftplusOpInputs {kData = 2}; 
*/

struct SoftplusParam : public dmlc::Parameter<SoftplusParam> { // Softplus层的参数说明, 结构体, 继承Parameter结构体.
  // Parameter结构体在dmlc-core/include/dmlc/parameter.h下114, SoftplusParam是结构体Parameter的模板参数. (模板参数是这样传递的)
   
  // 因为Softplus操作就一种激活函数, 为了保证和activation-inl.h一致, 也用枚举act_type来指定激活函数类型. 虽然act_type只有一种
  // 类型, 即值为0. 
  // 参数说明包括describe, set_default, set_range, add_num等. 
  int act_type;
  DMLC_DECLARE_PARAMETER(SoftplusParam) { // #define DMLC_DECLARE_PARAMETER(PType), 在这个宏下面定义了一些参数来对OP的参数进行
    // 说明, 设置等. 包括describe, set_default, set_range, add_num等.  
    DMLC_DECLARE_FIELD(act_type) // DMLC_DECLARE_FIELD宏. 传入OP的参数, 调用describe, set_default, set_range, add_num等.  
    .add_enum("softplus", softplus::kSoftplus) // 第一个参数字符串, 第二个参数是OP参数act_type的值. 
    .describe("Softplus activation function to be applied."); // 字符串. 
    /*
	DMLC_DECLARE_FIELD(act_type) 利用宏DMLC_DECLARE_FIELD对全连接层的参数act_type进行描述, 有些参数有默认值或最小, 最大值.  
	在描述该参数时, 可以对参数进行赋值. 
	*/
  }
};

/**
 * \brief This is the implementation of Softplus operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename ForwardOp, typename BackwardOp, typename DType> // 模板类. 
/*
xpu表示使用cpu函数gpu, ForwardOp表示前向操作, BackwardOp表示后向操作, DType表示数据类型.

在make mxnet的时候, 屏幕会根据config.mk给出xpu和DType的值. 
with [xpu = mshadow::op::cpu, DType = float] 或者 [with xpu = mshadow::cpu; DType = double]. 不同的机器上DType可能不一样. 

在使用ForwardOp和BackwardOp时, 利用的是./mshadow_op.h中自己定义的结构体. 下面会详细介绍. 
这里了保证和activation-inl.h一致, 也加了typename ForwardOp, typename BackwardOp这两个模板参数, 虽然Softplus前向和反向操作只有
一种运算. 
*/
class SoftplusOp : public Operator { // 定义SoftplusOp类, 继承Operator类. 定义Softplus操作的前向操作和后向操作. 
public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) { 
    /*前向操作, 虚函数. 函数的实现在类中定义. 不需要返回值. 
	OpContext: 结构体, 定义在include/mxnet/operator.h中(Dev的一个好处就是, 在当前打开的文档中, 利用ctrl+鼠标左键可以快速定位
	对象的定义, 某些极少的情况. 更多的可以利用Sublime的全局搜索功能), 该结构体可以记录操作在前向和后向传播中的信息. 
    
	in_data输入, 向量容器, 每个元素的类型是TBlob. Softplus层的输入数据只有data(上层的输出), 没有权重和偏置等. 因此in_data的大
	小为1. 只有kData. 
	 
	TBlob: mxnet底层数据格式引用了theano里面的Tensor和TBlob的形式, 使得代码清晰简洁. mxnet输入数据格式是TBlob, 
	输出是Tensor. tensor和TBlob是高维的数组类型. 下面会介绍Tensor烦人TBlo的联系和不同(简介). 
	
	req: 数据操作模式, 向量数组. 元素类型是OpReqType. 将计算的结果是如何写入到out_data(内存), 有的是直接写入, 有的是累加等.  
	OpReqType:  数据操作模式, kNullOp, kWriteTo, kWriteInplace, kAddTo. 以k开头, 后面英文的意思即操作模式.
	enum OpReqType{....} // 是一种枚举类型. include/mxnet/operator.h下面. 0, 1, 2, 3. 
	
	out_data: 输出, 向量容器, 每个元素的类型是TBlob. Sofyplus层输出数据out, 不输出其他的. 
	aux_args: 向量容器, 每个元素的类型是TBlob. 后面没用到.. 是为了方便计算而需要的附加的tensor, 现在是没有用到的. 
	这些辅助添加的Tensor目前只看到在Batch_Norm中使用. 
	
	in_data在这里不仅仅代表数据了, 这两个容器里面不仅有kData, 可能还有kWeight, kBias等等信息. 不同层的in_data和out_data
	是不一样的. in_data包含了上一层的输出做为本层的输入, 可能还有一些本层的参数信息(权重和偏置等); 利用out_data定义本层的输出.
	in_data 和 out_data 分别代表输入TBlob和输出TBlob. 所有的tensor需要的空间都是系统进行申请和管理.  
	*/
    using namespace mshadow; // 命名空间, mshadow用于数据存储结构. 
    using namespace mshadow::expr; // 表达式类. OP用到的一些表达式(函数)大多出自这个expr命名空间.   
    CHECK_EQ(in_data.size(), 1); // 判断输入数据的大小是否为1, verctor用.size访问其大小. 这里仅做检查, 不做输出. 
    CHECK_EQ(out_data.size(), 1); // 判断输出的大小是否为1, verctor用.size访问其大小. 这里仅做检查, 不做输出.
    /*
	检查宏, 如:
	CHECK, CHECK_EQ, CHECK_LT, 这些宏能够简化程序逻辑, 给Debug带来方便.
	*/
    Stream<xpu> *s = ctx.get_stream<xpu>(); 
    /*
	include/mxnet/operator.h下的get_stream()函数. 结构体OpContext成员ctx调用函数get_stream:
	template<typename xpu>
  	inline mshadow::Stream<xpu>* get_stream() const {...}. 获取指定Device(cpu, gpu)下的stream. 
	可以这样看, 在指定Device下定义OP的数据. 即数据是存储在cpu中还是gpu中. 
	 
	mshadow::Stream<xpu> *stream.  流其实是一种信息的转换, 是一个类的对象, 输入/输出流(I/O Streams). xpu表示是cpu还是gpu. 
	*/
	
	// std::cout<<"in_data kData: "<<softplus::kData<<std::endl; kData是0. 
    Tensor<xpu, 2, DType> data = in_data[softplus::kData].FlatTo2D<xpu, DType>(s);
    /*
	Tensor张量, 如矩阵就是二维的张量. 如, blob是caffe中的基本数据结构, 简单理解就是一个"4维数组", 
	caffe的blob就相当于一个特殊的tensor. Tensor张量就是多维数组.  Tesnor是一个模板结构体:
	struct Tensor: public TRValue<Tensor<Device, dimension, DType>, Device, dimension, DType>
	device表示设备, 即cpu还是gpu; dimension是张量的维数, 维数是2的张量就是矩阵; Dtype是张量元素的数据类型. 这里定义的data就是
	Dtype类型的矩阵. 
	
	Tensor的维数在定义后就固定了, 因此在图模型中需要一个更为抽象灵活的数据结构, 这就是TBlob.
	
	TBlob是mxnet的一种输入数据格式, 输出是Tensor. TBlob可以表示任意维数, 任意类型, 任意设备下的数据. TBlob与Tensor一样, 
	只是TBlob本身不包含算数运算, 当固定维数时TBlob就是Tensor了. TBlob是一个类. 
	
	TBlob不涉及任何算数运算, 也没有隐式的内存分配与释放, 它就像一个指针类, 在需要的时候调用get, get_with_shape, FlatTo2D
	FlatTo3D等获得固定维数的Tensor来做更多的操作. Tshape与TBlob类似, 在需要的时候调用get, FlatTo2D等获得Tensor对应的Shape.
	  
    in_data输入数据, 向量容器, 每个元素的类型是TBlob. 利用softplus::kData得到输入数据, 那么in_data[softplus::kData]就可以看成
	TBlob的对象, 因此可以调用TBlob下的函数FlatTo2D: ./include/mxnet/tensor_blob.h或./mshadow/mshadow/tensor_blob.h 
    inline mshadow::Tensor<Device, 2, DType> FlatTo2D(
          mshadow::Stream<Device> *stream = NULL) const {...}
	将in_data[0]拉成2维的张量, 即Tensor<xpu, 2, DType>. stream是目标流. .
	FlatTo2D<xpu, DType>是在使用函数FlatTo2D时指定device和type. 
	
	Tensor<xpu, 2, DType>和FlatTo2D<xpu, DType>, 即使用 类, 函数, 的时候指定模板参数:
	template<typename Device, int dimension, typename DType>和template<typename Device, typename DType>. 类, 函数定义时的模板
	参数是这样指定的. 
	
	这里指定定义类模板SoftplusOp(共有继承Operator操作类), 然后再利用:
    op = new SoftplusOp<cpu, mshadow_op::softplus, mshadow_op::softplus_grad, DType>();	// 使用模板参数. 
	来创建具体的类SoftplusOp. 在实例化SoftplusOp类时, mshadow_op::softplus, mshadow_op::softplus_grad就指定了前向和反向的具体
	实现.   
	*/
	
	// std::cout<<"out_data kOut: "<<softplus::kOut<<std::endl; kOut是0. 
    Tensor<xpu, 2, DType> out = out_data[softplus::kOut].FlatTo2D<xpu, DType>(s);
    /*
	与in_data的操作是一样的. 利用FlatTo2D将out_data[0]拉成2维的张量out. 
	这里in_data和out_data均是向量容器, 元素是TBlob的对象. 而softplus::kData和softplus::kOut可以看做是索引, 来指定容器中的数据
	成分. 定义Softplus的输出out. 
	
	这里只是将in_data和out_data中的输入和输入拉成2维的张量, 并没有涉及到激活函数的运算. 
	typename ForwardOp是前向操作, 如Relu函数激活;, typename BackwardOp是反向操作, 如Relu激活函数的梯度. 所以出现了ForwardOp
	和BackwardOp才涉及到真正的计算.  
	*/
	
	// std::cout<<"req kOut: "<<softplus::kOut<<std::endl; kOut是0. 
    Assign(out, req[softplus::kOut], F<ForwardOp>(data));
    /*
	赋值操作. Softplus输入是data, 输出是out, 给out赋值. 激活函数层并不包含任何的参数. 
	Assign操作定义在./operator_common.h 30行下, 是定义的一个宏函数. 根据需求将exp的值赋给out. 
	这不是C++的字符串赋值函数assign. 宏函数使用既有有点也有缺点:
	 
	#define Assign(out, req, exp)           
  	{                                     
	    switch (req) {                      
            case kNullOp:                     
               break;                          
            case kWriteTo:                    
            case kWriteInplace:               
                (out) = (exp);                  
                break;                          
            case kAddTo:                      
  		       (out) += (exp);                 
               break;                          
           default:                          
               LOG(FATAL) << "not reached";    
        }                                   
  	} 
   	req是向量容器, 元素的类型是OpReqType, 即是kNullOp, kWriteTo(直接写入), kWriteInplace(inplace write)
	kAddTo(表示应该调用+=, 例如将梯度累加起来, 而不是直接覆盖掉原来的结果). 
	还有一种赋值操作是, 宏ASSIGN_DISPATCH内部也对这4个类型进行了说明. include/mxnet/operator_util.h.
	
	softplus::kOut就是获取一种枚举类型(找到OpReqType类型的那个索引), 那么req[softplus::kOut]即OpReqType中的kWriteInplace或
	kAddTo, 然后通过exp给out赋值. 一般情况下, 所有的out_data的类型应该是kWriteTo, 表示out_data 表的tensor可以直接写入的原始的
	内存块. 在有些情况下, 比如说在计算gradient的tensor, 我们最好是将梯度累加起来, 而不是直接覆盖掉原来的结果, 这样我们就不需要
	每次计算的时候申请额外需要的内存空间. 在这种情况下, req的类型应该是kAddTo.
	
	F<ForwardOp>(data)即代表exp, 是表达式的值. ForwardOp即mshadow_op::softplus, 即利用激活函数Softplus执行操作:
	- `softplus`: `y = log(1 + exp(x))`.
	
	mshadow用于表达式操作的类(DotExp, BinaryMapExp, UnaryMapExp). 
	
	BinaryMapExp(二叉图)是双目运算的表达式类, 在BinaryMapExp类下有F函数, 
	F是自定义操作的函数, F< OP >(lhs, rhs)描述了一个新的双目运算(OP即ForwardOp和BackwardOp). 
	
	DotExp是做点乘的类, 其中最常用的就是dot函数.
	
	UnaryMapExp类(一元)是单目运算符的表达式类, 在UnaryMapExp类下有F函数:
    template<typename OP, typename TA, typename DType, int ta>
	inline UnaryMapExp<OP, TA, DType, (ta|type::kMapper)>
	    F(const Exp<TA, DType, ta> &src) {
  	    return MakeExp<OP>(src);
	} 
	
	这里F<ForwardOp>(data)是单目运算, 运算符是ForwardOp, 数据是data, 在Softplus前向后传播这是mshadow_op::softplus.  
	mshadow_op::softplus是定义在mshadow_op.h下的结构体: 传入a, 返回log(1 + exp(a)). 

	再将F<ForwardOp>(data)按某种数据操作模式写入方式赋值给out.  
	
	// 这里可以这样理解, 在python中, = 是浅拷贝, 即若x = y, 那么y改变时, x也随着改变.  
	*/
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    /*反向操作, 虚函数. 函数的实现在类中定义. 不需要返回值. 
	OpContext: 结构体. 只在前向和反向中使用.
	
	激活函数层没有权重W和偏置b, 因此也就没有损失L关于W和b的梯度. 即激活函数层只计算残差, 计算激活函数层的残差的时候需要用到上
	层的残差, sigma^(l) 和 sigma^(l + 1). 
	在不同的OP(层)中, in_grad和out_grad代表的含义是不同的. 在没有参数的层是残差(损失J关于输入Z的偏导), 有参数的层是梯度
	(损失关于权重W和偏置b的偏导). 这里先这样看, 以后遇到了问题再说明一下. 
	!!!!!!!!!!!!!!!!梯度可以看做是损失J关于层参数的导数, 残差可以看做是损失J关于层输入的导数!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
	  
	in_grad输出残差, 向量容器, 每个元素的类型是TBlob. 本层(第l层)的残差.
	out_grad输入残差, 向量容器, 每个元素的类型是TBlob. 上一层(第l + 1层)的残差, 计算本层的残差. 
	in_data输入, 向量容器, 每个元素的类型是TBlob. Softplus层的输入. 
	out_data输出, 向量容器, 每个元素的类型是TBlob. (第l层)的输出.	
	req: 数据操作模式, 向量数组. 元素类型是OpReqType. 
	*/
    using namespace mshadow; // 命名空间, mshadow用于数据存储结构. 
    using namespace mshadow::expr; // 表达式类. 
    CHECK_EQ(out_grad.size(), 1); // 上一层(第l + 1层)的残差容器大小是否为1. 即out_grad只有第l + 1层的残差. 
    CHECK(in_data.size() == 1 && in_grad.size() == 1); // 输入数据和输出梯度大小是否为1. 
    CHECK_EQ(req.size(), 1); // 数据操作模式是否一种. 
    Stream<xpu> *s = ctx.get_stream<xpu>(); // device流. 
    
    /*激活函数层的前向和反向传播:
	前向传播的过程就是: out = f(in). 再根据f的形式(ForwardOp)计算out即可.
	(例如mxnet FC层的反向传播和反向传播算法的梯度计算公式是一致的, 细节的地方还需要细心推敲).
	对于激活层的反向传播, 要计算其残差, 残差就是损失J关于本层输入Z的偏导:
	in_grad = partial(J) / partial(Z^(l)) =(链式法则)partial(J) / partial(Z^(l + 1)) * partial(Z^(l + 1)) / partial(Z^(l))
	这里, partial(J) / partial(Z^(l + 1))即为上一层的残差, 即out_grad[0]. 
	Z^(l + 1) = f(Z^(l))(激活函数层的输入是Z^(l), 输出是 f(Z^(l)), f(Z^(l))做为下一层的输入), 
	因此partial(Z^(l + 1)) / partial(Z^(l)) = f'(Z^(l)). 这样就正好用到了BackwardOp, 例如sigmoid_grad:
		
    f'(x) = f(x) * (1 - f(x)), 因此f'(Z^(l)) =  f(Z^(l)) * (1 - f(Z^(l)) = Z^(l + 1) * (1 - Z^(l + 1)).
	第l + 1层的输入 * (1 - 第l + 1层的输入). 第l + 1层的输入即本层(第l层)的输出, 即out_data[0]. 
	因此,  f'(Z^(l)) = a^(l) * (1 - a^(l)). 
    
    而对于Softplus函数来说, f(x) = ln(1 + exp(x)), 因此f'(Z^(l)) = [exp(Z^(l))] / [1 + exp(Z^(l))], 又
	Z^(l + 1) = ln(1 + exp(Z^(l))), 因此, exp(Z^(l + 1)) = 1 + exp(Z^(l)). 故f'(Z^(l)) = 1 - exp( - Z^(l + 1)).
	
    在激活函数层的反向传播中要计算的是f'(Z^(l)), 然后化成关于Z^(l + 1)的式子. 并不是计算f'(x)就行了, 还需要将x换为Z^(l), 再
	计算成关于Z^(l + 1)的式子. 即为BackwardOp所做的事. 
	BackwardOp即接收Softplus的输出 Z^(l), 然后计算 f'(Z^(l)) = 1 - exp( - Z^(l + 1)).  
	有了f'(Z^(l)), 就可以计算Softplus损失关于本层输入的残差了! 
	
	即本激活函数层(第l层)的残差就是:
    F<BackwardOp>(m_out_data) * m_out_grad. m_out_grad上一层(l + 1)的梯度, m_out_data本层的输出. 
	*/
	
	// std::cout<<"out_grad kOut: "<<softplus::kOut<<std::endl; kOut是0. 
	Tensor<xpu, 2, DType> m_out_grad = out_grad[softplus::kOut].FlatTo2D<xpu, DType>(s);
    /*
	利用FlatTo2D, 将上层(l + 1)的残差out_grad[0]拉成2维的张量, 即维数固定后就是Tensor了. 赋给m_out_grad. 
	*/
	
	// std::cout<<"in_grad kData: "<<softplus::kOut<<std::endl;; kOut是0. 
    Tensor<xpu, 2, DType> m_out_data = out_data[softplus::kOut].FlatTo2D<xpu, DType>(s);
    /*
	利用FlatTo2D, 将本层(l层)的输出a^(l)即out_data[0]拉成2维的张量, 即维数固定后就是Tensor了. 赋给m_out_data.  
	*/
	
	// std::cout<<"in_grad kData: "<<softplus::kOut<<std::endl; kData是0. 
    Tensor<xpu, 2, DType> m_in_grad = in_grad[softplus::kData].FlatTo2D<xpu, DType>(s);
    /*
	定义本层(第l层)的残差, 本层的残差m_in_grad也是2维的张量. 
	
	前向传播中的out_data和反向传播中的in_grad, 可以这样理解, 先申请好Softplus输出和残差的内存空间, 然后再将数据写入内存即可. 
	*/
	
	// std::cout<<"req kData: "<<softplus::kData<<std::endl; kData是0. 
    Assign(m_in_grad, req[softplus::kData], F<BackwardOp>(m_out_data) * m_out_grad);
    /*
	根据上述分析的激活函数层的反向传播推导, 激活函数层的反向传播即算本层(第l层)的残差:
    m_in_grad = F<BackwardOp>(m_out_data) * m_out_grad. 
	
	该句是赋值语句, 通过req[softplus::kData](某一种数据操作模式, kWriteTo)将F<BackwardOp>(m_out_data) * m_out_grad写入
	m_in_grad.
	
	F<BackwardOp>(m_out_data)利用反向传播模板BackwardOp和本层(第l层)的输出, 计算偏导数. 例如对Sigmoid函数来说, 
	F<BackwardOp>(m_out_data)就是: m_out_data * (1 - m_out_data).
	对于Softplus, F<BackwardOp>(m_out_data)就是 1 - (expf( - m_out_data)). 即 f'(Z^(l)). 
	
	想做激活函数层的前向和反向传播, 只需在mshadow_op.h定义好前向传播ForwardOp和反向传播BackwardOp即可. 
	
	对于mxne的其他层来说, 前向和反向传播可能就只有一种情况, 因此前向和反向操作就用不到前向传播ForwardOp和反向传播BackwardOp了. 
	*/
  }
};  // class SoftplusOp 


// Decalre Factory function, used for dispatch specialization
template<typename xpu> // mxnet的device都定义成模板参数xpu. 
Operator* CreateOp(SoftplusParam type, int dtype);
/*类Operator在include/mxnet/operator.h下, 定义操作基类(operator)等.
创建OP操作, 在softplus.cc中用到. 
和op = new SoftplusOp<cpu, mshadow_op::softplus, mshadow_op::softplus_grad, DType>(); 是相连系的, 即调用函数CreateOp来创建OP类,
op即表示了mxnet的layer. 

创建op函数仅在-inl.h中声明, 具体定义在.cc中. CreateOp建立OP, 传入SoftplusParam的参数对象type和dtype. 在Softplus.cc中会介绍.  
*/

#if DMLC_USE_CXX11 // C++11. 下面是激活函数层Softplus的属性, 每个OP的属性可能均不一样, 有些属性也是可选的. 
// 一般也就需要infer_shape, 和OP输入参数和输出参数的描述. 
class SoftplusProp : public OperatorProperty { // 类SoftplusProp, 继承OperatorProperty(操作属性), 在include/mxnet/operator.h下
/*在SoftplusProp类中的方法均重写了父类OperatorProperty的方法. 

override是一个关键字, 和overload类似. 在类中使用关键字override或overload是表明该函数是重写还是重载.
重载(overload)某个方法是在同一个类中发生的, 重写(override)是在子类中重写父类中的方法. 

父类OperatorProperty在定义时, 声明了很多OP属性的函数, 具体到某一个OP时, 这些属性函数会有区别, 因此要对父类的这些属性函数进行
重写. 父类OperatorProperty定义的方法均有自己的实现主体, 计算在定义新op时不重写这些方法也可以用. 
*/
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }
  /*重写父类OperatorProperty的Init函数
  Init函数在所有的OP中均会用到, 通过设置参数kwargs来初始化OP. param_是SoftplusParam定义的私有成员, 来调用Init函数.
  
  Parameter结构体在dmlc-core/include/dmlc/parameter.h下114, 由于结构体SoftplusParam继承了Parameter结构体, 因此SoftplusParam
  的成员param_可以调用Parameter结构体的函数 Init:
  template<typename Container>
  inline void Init(const Container &kwargs, parameter::ParamInitOption option = parameter::kAllowHidden) {...}
  用来根据kwargs初始化参数的, 即初始化结构体SoftplusParam的参数. 
  param kwargs map of keyword arguments, or vector of pairs. 
  
  pair是一种模板类型, 其中包含两个数据, 如pair<string, string> a("James", "Joy");
  kwargs是向量容器, 每一个元素均为一种模板类型, 可以保存一对值.   
  */

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  } 
  /*固定格式. 获取参数, 重写父类OperatorProperty的GetParams函数.
  
  用来获取OP的内部参数, 如act_type等. 将结果保存在map容器中. 
  param_是SoftplusParam的对象, 由于结构体SoftplusParam继承了Parameter结构体, 因此SoftplusParam的成员param_可以调用
  Parameter结构体的函数 __DICT__:
  inline std::map<std::string, std::string> __DICT__() const {...}. 将OP的参数放到了一个dict里, 即key -> value. 是一对值. 
  */ 
  
  /*另外父类OperatorProperty中还有函数ListArguments和ListOutputs, 用来指定op的输入参数和输出参数. 不同的op, ListArguments和
  ListOutputs也是有所区别的.  
  */

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    /*重写父类OperatorProperty的InferShape(shape推断)函数.
  	这个接口有两个目的: (1)向系统提供每个输入Tensor和输出Tensor的大小(形状), 这样系统可以在进行Forward和Backward之前申请好相应
  	的内存. 即向上面提到的 out_data容器 和 in_grad容器, 根据提供的Tensor的大小, 首先先申请内存, 然后再做赋值操作; 
	(2)进行类型检查, 在运行前确保没有明显的错误. in_shape中的shape是系统自动设置(依赖上个Operator的out_shape ). 
  	如果系统认为提供的信息不足以完成shape的推断会返回false, 或者在shape不一致的时候抛出异常.
  	*/
	/*
	TShape: 一个shape类, 该类可以表示一个Tensor的形状. 利用TShape来表示Tensor的大小. 2维张量即矩阵, 即矩阵的大小.
	
	in_shape和out_shape表示输入和输出Tensor的shape, 向量容器(指针类型的向量容器), 元素的类型是TShape的对象:
	例如: in_shape[0]代表[64, 100]. out_shape可以通过in_shape来定义.
	
	aux_shape和上文的aux_args一样, 是想附加一个tensor, 暂未用到. 
	*/ 
    using namespace mshadow; // 命名空间. 其下定义了很多类和方法. 
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]"; // 检查in_shape容器的大小是否为1, 若不是做输出. 
    
    // std::cout<<"in_shape kData: "<<softplus::kData<<std::endl; kData是0. 
    const TShape &dshape = in_shape->at(softplus::kData);
    // std::cout<<"dshape: "<<dshape<<std::endl; dshape是(64,128), 64是batch_size即批训练数目的大小, 128是前一层输出的数据的
	// 个数, 在实验中Softplus前一层是FC层, FC层有128个结点, 因此这里也是128. 
    
    /*
	定义dshape, 常TShape引用. 通过输入tensor的形状对dshape进行赋值, 由于in_shape是向量容器, 这里是有向量容器的at函数.
	at()函数, 用来访问vector指定位置loc的元素. 和in_shape[i]有相同的作用. 
	at()函数比[]运算符更加安全, 因为它不会让你去访问到Vector内越界的元素. 
	
	in_shape->at(activation::kData)即是访问data_shape, 利用枚举类型activation::kData访问data_shape, 即in_shape[0].
	
	mxnet中, vector向量容器调用容器函数均是用->完成的(in_shape->at( )), 平时用的是 . 即in_shape.at() . 
	*/
    
    if (dshape.ndim() == 0) return false; // ndim表示维数, 代表张量Tensor的维数, 例如2. 
    out_shape->clear(); // 清空out_shape, 以存入新的shape. 
    out_shape->push_back(dshape); // 如果tensor的维数不是0, 那么就将data_shape存入out_shape中, 即输入和输出是一样的. 
	/*
	因为在定义容器的时候定义的指针类型的容器, 所以在使用容器的成员函数时, 操作符是->, 而不是 .  
	*/ 
    return true;
    /*
	如果shape推断成功, 返回true; 如果没有足够的信息来进行shape推断, 则返回false; 如果输入和输出不一致, 抛出异常. 
	*/
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    /*重写父类OperatorProperty的InferType(类型推断)函数. 
  	推断输出的数据类型和未知的输入参数. 
  
   	in_type: int型的向量容器. 输入参数的类型.
  	out_type: 输出的类型. (指针类型的向量容器) 
  	aux_type: 辅助状态的类型, 暂未用到. 
  	*/				 				  
				 				  
    CHECK_GE(in_type->size(), 1); // in_type容器的大小是否大于等于1. 
	/*
	1.日志输出宏:
    LOG(WARNING) << "This is a warning message";
    2.CHECK_XX宏:
    1 #define CHECK_EQ(val1, val2) CHECK_OP(_EQ, ==, val1, val2)
	2 #define CHECK_NE(val1, val2) CHECK_OP(_NE, !=, val1, val2)
	3 #define CHECK_LE(val1, val2) CHECK_OP(_LE, <=, val1, val2)
	4 #define CHECK_LT(val1, val2) CHECK_OP(_LT, < , val1, val2)
	5 #define CHECK_GE(val1, val2) CHECK_OP(_GE, >=, val1, val2)
	6 #define CHECK_GT(val1, val2) CHECK_OP(_GT, > , val1, val2) 
	*/ 
    int dtype = (*in_type)[0]; // 初始化dtype, 类型使用int型的变量来表示的. (*in_type)[0]是int型的变量, *in_type是一个指针型的
	// 容器, 因此(*in_type)[0]就表示这个指针型容器的第一个分量. 是一个int型的变量. 
	// std::cout<<"dtype: "<<dtype<<std::endl; dtype是0. 
	
    CHECK_NE(dtype, -1) << "First input must have specified type"; // 判断dtype是否和-1相等, 如果相等, 输出信息. 
    
    for (index_t i = 0; i < in_type->size(); ++i) { // index_t是一种索引类型, 通常是无符号的.  
      /*
      index_t的定义在mshadow/mshadow/base.h下.
	  typedef unsigned index_t; 
	  unsigned a; 与unsigned int a; 是同样的. 这里省略了int, 只能和unsigned int等价.
	  
	  指针型容器引用一些成员函数时用 in_type->, 访问容器元素时用 in_type[i]. i 即要访问 in_type 容器的每一个元素. 
	  */
      if ((*in_type)[i] == -1) { // 判断(*in_type)[i]是否为-1. 
          (*in_type)[i] = dtype; // (*in_type)[i]为-1, 则(*in_type)[i]为dtype. 
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
        // 判断(*in_type)[i]是否和dtype相等. 如果不相等输出信息. ListArguments()是罗列参数的. 
      }
    }
    out_type->clear();
    out_type->push_back(dtype); // 将dtype即(*in_type)[0]赋值给out_type, 因此输入和输出的类型也是一样的. 
    return true;
  }

  OperatorProperty* Copy() const override {
    /*重写父类OperatorProperty的Copy函数. 该函数会copySoftplusProp(操作属性类)的参数. 
    */
    auto ptr = new SoftplusProp(); 
    /*
	自动推断类型auto. 创建SoftplusProp的对象, 并调用SoftplusProp类的构造函数(默认构造函数)初始化对象ptr.
	类名 对象名 =new 类名(); // 调用 类/结构体 定义 + 初始化一个对象. 
	*/
    ptr->param_ = param_; // SoftplusProp类的对象ptr调用.... 
    return ptr;
  }

  std::string TypeString() const override {
    return "Softplus";
  }
  /*重写父类OperatorProperty的TypeString函数. 指定该op的名称. 
  */

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
#if MXNET_USE_CUDNN == 1
    return {out_grad[softplus::kOut], out_data[softplus::kOut], in_data[softplus::kData]};
#else
    return {out_grad[softplus::kOut], out_data[softplus::kOut]}; // 返回列表, 根据是否使用CUDNN(GPU)返回不同的列表. 
#endif  // MXNET_USE_CUDNN
  }
  /*重写父类OperatorProperty的DeclareBackwardDependency函数. 该函数在反向传播时声明input requirement.
  返回在反向传播时用到的列表, 这个函数主要是用来优化内存的. 有时候tensor在做反向传播时不需要了, 就要释放, 这就是垃圾回收机制.
  在定义新的op时, 该函数需要重写, 来指定在反向过程中到底需要哪些变量. 如果不重写, 默认的DeclareBackwardDependency将清空所有的
  变量.
  
  out_grad: 在反向传播中输出的梯度(残差). 上一层(第l + 1层)的梯度/残差. 
  in_data: 前向过程中的Softplus层的输入数据.
  out_data: 前向过程中的Softplus层的输出数据. 
  
  */

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[softplus::kOut], in_grad[softplus::kData]}};
  }
  /*重写父类OperatorProperty的BackwardInplaceOption函数. 为了进一步的节省内存的申请开销, 我们倾向于是用原地更新(inplace update). .
  这个主要用在element-wise操作上, 因为这种情况下输入tensor和输出tensor的shape是一致的. 
  
  out_grad[0]和in_grad[0]分享同样的内存空间在Backward计算过程中. 
  
  out_grad: 在反向传播中输出的梯度(残差).
  in_data: 前向过程中的输入.
  out_data: 前向过程中的输出. 
  in_grad: 反向中的输入梯度(残差). 
  
  pair模板类, 一对值. 
  
  */

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[softplus::kData], out_data[softplus::kOut]}};
  }
  /*重写父类OperatorProperty的ForwardInplaceOption函数. 该函数和BackwardInplaceOption的作用是一样的.
  
  in_data[0]和out_data[0]的tensors应该在Forward的计算过程中使用同样的内存空间.
  
  in_data: 前向过程中的输入.
  out_data: 前向过程中的输出.  
  */

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented."; // 日志的级别是 FATAL(致命的). 
    return NULL;
  }
  /*重写父类OperatorProperty的CreateOperator函数.固定格式. Create a Operator on specific context.
  */

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;
  /*重写父类OperatorProperty的CreateOperatorEx函数.固定格式. 
  Create a Operator on specific context and input shape/type.
  
  -inl.h中只是对重写函数进行了声明, 该函数的实现在.cc中. .cc文件include-inl.h文件. 
  */
  
  /*
  有的层可能还需要重写ForwardResource和BackwardResource函数:
  有些操作需要额外的内存作为工作空间来进行计算, 比如说cudnnConvolutionForward. 这种情况下, 系统最好可以对这部分内存进行管理, 
  这样系统可以做一些优化, 比如说内存的重复利用. MXNet定义了两个接口来达到目的: ForwardResource和BackwardResource函数.
  */

 private:
  SoftplusParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SOFTPLUS_INL_H_
