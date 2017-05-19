/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling.cc
 * \brief
 * \author Bing Xu
*/
#include "./pooling-inl.h"
#if MXNET_USE_MKL2017 == 1 // 如果使用MKL2017. 就采用src/operator/mkl下的pooling操作来定义操作Pooling. 
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_pooling-inl.h"
#endif  // MXNET_USE_MKL2017
#if MXNET_USE_NNPACK == 1 // 如果使用NNPACK, 就利用src/operator/nnpack下的pooling操作来定义操作Pooling. 
#include "./nnpack/nnpack_pooling-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(PoolingParam param, int dtype) { // 创建op, 传入参数param和dtype, dtype是(*in_type)[0], 暂未用到. 
  Operator *op = NULL; // 声明新的op为空, 然后通过-inl.h中定义好的类SoftplusOp来定义op. 
#if MXNET_USE_MKL2017 == 1 // 使用MKL2017. 
    if ((param.pool_type == pool_enum::kMaxPooling)
      || (param.pool_type == pool_enum::kAvgPooling
      && UseMKLPooling(param))) { // 池化类型是最大池化或平均池化.  
      switch (dtype) { // dtype是一个int型的变量, 代表是kFloat32还是kFloat64. 
      case mshadow::kFloat32:
        return new MKLPoolingOp<cpu, float>(param); // 创建池化OP. 和MKLPoolingOp的类定义有关.  
      case mshadow::kFloat64:
        return new MKLPoolingOp<cpu, double>(param);
      default:
        break;
      }
    }
    LOG(INFO) << MKLPoolingOp<cpu, float>::getName() << " Skip MKL optimization";
#endif
#if MXNET_USE_NNPACK == 1 // 使用NNPACK. 
  // NNPACK only support max-pooling with kernel = 2, stride = 2, pooling_convention
  // = kFull(note that the default value is kValid in MXNet)
  if ((param.pool_type == pool_enum::kMaxPooling) // 最大池化操作.  
    && (param.pooling_convention == pool_enum::kFull)
    && (param.kernel.ndim() == 2) && (param.stride.ndim() == 2)
    && (param.kernel[0] == 2) && (param.kernel[1] == 2)
    && (param.stride[0] == 2) && (param.stride[1] == 2)) {
    switch (dtype) {
    case mshadow::kFloat32:
      return new NNPACKPoolingOp<cpu, mshadow::red::maximum, float>(param);
    default:
      break;
    }
  }
#endif
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { // 不使用MKL2017和NNPACK的情况下穿件池化操作! 
    switch (param.pool_type) { // param是PoolingParam的对象, 访问pool_type, 得到kMaxPooling或kAvgPooling或kSumPooling. 
      case pool_enum::kMaxPooling:
        op = new PoolingOp<cpu, mshadow::red::maximum, DType>(param); // 创建最大池化. 利用new创建. 
        break;
      case pool_enum::kAvgPooling:
        op = new PoolingOp<cpu, mshadow::red::sum, DType>(param);
        break;
      case pool_enum::kSumPooling:
        op = new PoolingOp<cpu, mshadow::red::sum, DType>(param);
        break;
      default:
        LOG(FATAL) << "unknown pooling type";
        return NULL;
    }
    /*
	在定义池化操作类PoolingOp时, 定义的是模板类. 模板参数是:
	template<typename xpu, typename Reducer, typename DType> 
	
	在利用new创建池化操作时, 会指定xpu和Reducer, DType在编译时指定.  
	*/
  })

  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* PoolingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

/*
使用下面的宏定义来将parameter结构和OperatorProperty类注册到MXNet系统中:
DMLC_REGISTER_PARAMETER和MXNET_REGISTER_OP_PROPERTY. 
*/
DMLC_REGISTER_PARAMETER(PoolingParam);

MXNET_REGISTER_OP_PROPERTY(Pooling, PoolingProp)
.describe("Perform spatial pooling on inputs.")
.add_argument("data", "Symbol", "Input data to the pooling operator.")
.add_arguments(PoolingParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
