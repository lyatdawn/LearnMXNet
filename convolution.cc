/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cc
 * \brief
 * \author Bing Xu
*/

#include "./convolution-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_convolution-inl.h"
#endif  // MXNET_USE_MKL2017 是否使用MKL2017. 
#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_convolution-inl.h"
#endif  // MXNET_USE_NNPACK // 是否使用NNPACK. NNPACK库只做了 卷积, FC的. 

namespace mxnet {
namespace op {		  
/*
使用下面的宏定义来将parameter结构和OperatorProperty类注册到MXNet系统中:
DMLC_REGISTER_PARAMETER和MXNET_REGISTER_OP_PROPERTY. 
*/
DMLC_REGISTER_PARAMETER(ConvolutionParam);

template<>
Operator* CreateOp<cpu>(ConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) { // 创建op, 传入参数param和dtype, in_shape(指针型容器), out_shape. 
  // 是convolution-inl.h中类: 
  /*
  template<typename xpu>
  Operator* CreateOp(ConvolutionParam param, int dtype,
                   std::vector<TShape> *in_shape,
                   std::vector<TShape> *out_shape,
                   Context ctx);
  */ 
  Operator *op = NULL; // 卷积操作 op 初始化为空. 
#if MXNET_USE_MKL2017 == 1
  if ((param.dilate[0] == 1 && param.dilate[1] == 1)
      && param.kernel.ndim() == 2) {
    switch (dtype) {
    case mshadow::kFloat32:
      return new MKLConvolutionOp<cpu, float>(param);
    case mshadow::kFloat64:
      return new MKLConvolutionOp<cpu, double>(param);
    default:
      break;
    }
  }
  LOG(INFO) << MKLConvolutionOp<cpu, float>::getName() << " Skip MKL optimization";
#endif // 使用MKL2017. 
#if MXNET_USE_NNPACK == 1
  const size_t batch_size = (*in_shape)[0][0];
  if ((param.dilate[0] == 1 && param.dilate[1] == 1)
      && param.kernel.ndim() == 2 && (!param.no_bias)
      && param.num_group == 1 && (batch_size == 1 ||
      ((batch_size > 1) && (param.stride[0] == 1) &&
      (param.stride[1] == 1)))) {
    switch (dtype) {
    case mshadow::kFloat32:
      return new NNPACKConvolutionOp<cpu, float>(param);
    default:
      break;
    }
  }
#endif // 使用NNAPCK. 
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { // 不使用MKL2017和NNPACK库, 定义卷积操作对象op. 
    op = new ConvolutionOp<cpu, DType>(param);
    /*
	调用ConvolutionOp类的构造函数:
	explicit ConvolutionOp(ConvolutionParam p) {...}. 传入ConvolutionParam的对象param, 创建ConvolutionOp类的对象op. 即
	op就可以代表卷积操作.  
	*/
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ConvolutionProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionProp)
.add_argument("data", "Symbol", "Input data to the ConvolutionOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.") // 调用add_argument, 给BN层添加参数.
.add_arguments(ConvolutionParam::__FIELDS__())
.describe("Apply convolution to input then add a bias."); // 调用describe函数描述

}  // namespace op
}  // namespace mxnet
