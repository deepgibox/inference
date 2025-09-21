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
    
    // -------- Memory allocation --------
    // Input allocator on CUDA device (for fast inference)
    let input_allocator = Allocator::new(
        &session,
        MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
    )?;
    
    // Output allocator on CPU (so we can read the results)
    let cpu_allocator = Allocator::new(
        &session,
        MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?,
    )?;
    
    // Allocate input tensor on GPU (1,3,512,512)
    let input_tensor = Tensor::<f32>::new(&input_allocator, [1_usize, 3, 512, 512])?;
    
    // Allocate output tensor on CPU (so we can read it directly)
    let output_tensor = Tensor::<f32>::new(&cpu_allocator, [1_usize, 16128, 7])?;
    
    // -------- IOBinding --------
    let mut io_binding = session.create_binding()?;
    io_binding.bind_input("images", &input_tensor)?;   // GPU input
    io_binding.bind_output("output", output_tensor)?;  // CPU output
    
    // TODO: Fill input_tensor with your actual image data here
    // For now, it contains zeros which won't give meaningful YOLO results
    
    // Run inference once
    let start = Instant::now();
    let outputs = session.run_binding(&io_binding)?;
    let duration = start.elapsed();
    
    println!("Inference duration: {:?}", duration);
    println!("Inference done with IOBinding!");
    println!("Output tensors count: {}", outputs.len());
    
    // Extract and read the CPU output
    match outputs[0].try_extract_tensor::<f32>() {
        Ok((shape, data)) => {
            println!("✓ Output ready on CPU!");
            println!("Output shape: {:?}", shape);
            println!("Total elements: {}", data.len());
            
            // Show first 10 values
            println!("First 10 output values:");
            for (i, val) in data.iter().take(10).enumerate() {
                println!("  output[{i}] = {val}");
            }
            
            // Process YOLO detections (example)
            if shape.len() >= 3 {
                let batch_size = shape[0];
                let num_detections = shape[1]; 
                let values_per_detection = shape[2];
                
                println!("\nYOLO Output Analysis:");
                println!("  Batch size: {}", batch_size);
                println!("  Number of detections: {}", num_detections);
                println!("  Values per detection: {}", values_per_detection);
                
                // Show first few detections
                println!("\nFirst 5 detections:");
                for detection_idx in 0..5.min(num_detections as usize) {
                    let start_idx = detection_idx * values_per_detection as usize;
                    let end_idx = start_idx + values_per_detection as usize;
                    let detection = &data[start_idx..end_idx];
                    println!("  Detection {}: {:?}", detection_idx, detection);
                }
            }
        },
        Err(e) => {
            println!("Error extracting tensor data: {:?}", e);
        }
    }
    
    Ok(())
}