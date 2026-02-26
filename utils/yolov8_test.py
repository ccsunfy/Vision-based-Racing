# #####from pt to onnx
# from ultralytics import YOLO # 1️⃣ 加载 YOLOv8n 预训练模型
# model = YOLO("yolov8n.pt") # 2️⃣ 导出为 ONNX（动态 batch-size） 
# model.export(format="onnx", dynamic=True)


######from onnx to simplified onnx
# import onnx
# import onnxsim
 
# # 1️⃣ 加载 ONNX 模型
# onnx_model = onnx.load("yolov8n.onnx")
 
# # 2️⃣ 进行模型简化
# simplified_model, check = onnxsim.simplify(onnx_model)
 
# # 3️⃣ 保存优化后的模型
# onnx.save(simplified_model, "yolov8n_sim.onnx")
# print("ONNX Simplified:", check)

#####from onnx to tensorrt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

def build_engine(onnx_path, engine_path, precision_mode="fp16", max_batch_size=1, max_workspace_size=1 << 30):
    """
    将ONNX模型转换为TensorRT引擎
    
    参数:
    - onnx_path: ONNX模型文件路径
    - engine_path: 保存TensorRT引擎的路径
    - precision_mode: 精度模式 ("fp32", "fp16", "int8")
    - max_batch_size: 最大批处理大小
    - max_workspace_size: 最大工作空间大小(字节)
    """
    # 初始化TensorRT日志记录器和构建器
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 配置构建器
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    
    # 设置精度模式
    if precision_mode == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision_mode == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # 对于INT8模式，通常需要设置校准器
        # config.int8_calibrator = MyCalibrator(calibration_data)
    
    # 解析ONNX模型
    print(f"加载ONNX模型: {onnx_path}")
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("解析失败:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print("ONNX模型解析成功!")
    print(f"网络层数: {network.num_layers}")
    
    # 构建引擎
    print(f"构建TensorRT引擎 (精度: {precision_mode})...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("引擎构建失败!")
        return None
    
    # 保存引擎
    print(f"保存引擎到: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    
    print("转换完成!")
    return engine

def load_engine(engine_path):
    """从文件加载TensorRT引擎"""
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    
    return runtime.deserialize_cuda_engine(engine_data)

def inference(engine, input_data):
    """使用TensorRT引擎进行推理"""
    # 创建执行上下文
    context = engine.create_execution_context()
    
    # 分配输入/输出缓冲区
    bindings = []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # 分配设备内存
        device_mem = cuda.mem_alloc(size * dtype.itemsize)
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(binding):
            print(f"输入: {binding}, 形状: {engine.get_binding_shape(binding)}, 类型: {dtype}")
            # 复制输入数据到设备
            cuda.memcpy_htod(device_mem, input_data.ravel())
    
    # 执行推理
    context.execute_v2(bindings)
    
    # 获取输出
    outputs = []
    for binding in engine:
        if not engine.binding_is_input(binding):
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            shape = (engine.max_batch_size,) + engine.get_binding_shape(binding)
            size = trt.volume(shape) * dtype.itemsize
            host_mem = cuda.pagelocked_empty(shape, dtype)
            # 从设备复制数据到主机
            cuda.memcpy_dtoh(host_mem, bindings[engine.get_binding_index(binding)])
            outputs.append(host_mem)
    
    return outputs

if __name__ == "__main__":
    ONNX_PATH = "yolov8n_sim.onnx"
    ENGINE_PATH = "yolov8.engine"
    PRECISION = "fp16"  # 可选: "fp32", "fp16", "int8"
    
    # 步骤1: 转换ONNX到TensorRT引擎
    build_engine(
        onnx_path=ONNX_PATH,
        engine_path=ENGINE_PATH,
        precision_mode=PRECISION,
        max_batch_size=1,
        max_workspace_size=1 << 30  # 1GB
    )
    
    # 步骤2: 加载引擎并进行推理
    engine = load_engine(ENGINE_PATH)
    
    # 创建示例输入数据 (根据实际模型调整)
    import numpy as np
    input_shape = (1, 3, 224, 224)  # 示例输入形状: batch, channels, height, width
    dummy_input = np.random.random(input_shape).astype(np.float32)
    
    # 执行推理
    outputs = inference(engine, dummy_input)
    
    print(f"推理完成! 输出形状: {outputs[0].shape}")