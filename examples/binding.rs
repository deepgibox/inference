use ort::{
    session::Session,
    execution_providers::TensorRTExecutionProvider,
    memory::{Allocator, MemoryInfo, AllocationDevice, AllocatorType, MemoryType},
    value::Tensor,
    Result,
};
use std::time::Instant;

fn main() -> Result<()> {
    // Init environment with TensorRT EP
    let _env = ort::init()
        .with_name("tensorrt_iobinding")
        .with_execution_providers([
            TensorRTExecutionProvider::default()
                .with_device_id(0)
                .with_fp16(true)
                .with_engine_cache(true)
                .with_engine_cache_path("./trt_cache")
                .build(),
        ])
        .commit();

    // Load model
    let model_path = "./YOLOv5.onnx";
    let mut session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)?;

    println!("Session ready with TensorRT!");

    // -------- GPU memory allocation --------
    // Input allocator on CUDA device
    let input_allocator = Allocator::new(
        &session,
        MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
    )?;

    // Output allocator on CUDA device
    let output_allocator = Allocator::new(
        &session,
        MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
    )?;

    // Allocate input tensor directly on GPU (1,3,512,512)
    let input_tensor = Tensor::<f32>::new(&input_allocator, [1_usize, 3, 512, 512])?;

    // Allocate output tensor directly on GPU (adjust shape to your model)
    let output_tensor = Tensor::<f32>::new(&output_allocator, [1_usize, 16128, 7])?;

    // -------- IOBinding --------
    let mut io_binding = session.create_binding()?;
    io_binding.bind_input("images", &input_tensor)?;   // input name from model
    io_binding.bind_output("output", output_tensor)?; // output name from model

    // Run with binding (everything stays on GPU)
    let start = Instant::now();
    let outputs = session.run_binding(&io_binding)?;
    let duration = start.elapsed();

    println!("Inference duration: {:?}", duration);
    print!("Inference done with IOBinding!");

    Ok(())
}
