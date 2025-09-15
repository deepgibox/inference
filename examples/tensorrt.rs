use ort::{
    session::Session, 
    value::Tensor, 
    execution_providers::TensorRTExecutionProvider,
    Result
};
use ndarray::Array4;

fn main() -> Result<()> {
    // Example with advanced TensorRT configuration
    let _committed = ort::init()
        .with_name("advanced_tensorrt_env")
        .with_telemetry(false)
        .with_execution_providers([
            // Advanced TensorRT configuration
            TensorRTExecutionProvider::default()
                .with_device_id(0)                    // Use GPU 0
                .with_fp16(true)                      // Enable FP16 precision
                .with_int8(false)                     // Disable INT8 for now
                .with_max_workspace_size(2 << 30)     // 2GB workspace for large models
                .with_min_subgraph_size(3)            // Minimum 3 nodes for TensorRT optimization
                .with_max_partition_iterations(1000)  // More partition iterations
                .with_engine_cache(true)              // Enable engine caching
                .with_engine_cache_path("./trt_engines") // Custom cache directory
                .with_engine_cache_prefix("yolo_")    // Prefix for cache files
                .with_dump_subgraphs(true)            // Debug: dump subgraphs
                .build()
        ])
        .commit();

    println!("TensorRT environment initialized: {}", _committed);

    // Use a test model - in practice, replace with your actual model path
    let model_path = "./YOLOv5.onnx";
    
    println!("Loading model: {}", model_path);
    
    let mut session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file(model_path)?;

    println!("TensorRT session created successfully!");
    
    // Print model information
    println!("\nModel inputs:");
    for (i, input) in session.inputs.iter().enumerate() {
        println!("  Input {}: {} ({:?})", i, input.name, input.input_type);
    }
    
    println!("\nModel outputs:");
    for (i, output) in session.outputs.iter().enumerate() {
        println!("  Output {}: {} ({:?})", i, output.name, output.output_type);
    }

    // Create sample input (adjust shape based on your model)
    let input_shape = (1, 3, 512, 512); 
    let input_data: Array4<f32> = Array4::ones(input_shape) * 0.5; 
    
    let input_tensor = Tensor::from_array(input_data)?;
    
    println!("\nRunning inference with TensorRT...");
    
    let start = std::time::Instant::now();
    let outputs = session.run(ort::inputs![input_tensor.clone()])?;
    let first_run_time = start.elapsed();
    
    println!("First inference completed in: {:?}", first_run_time);
    println!("Number of outputs: {}", outputs.len());
    
    for (i, (name, output)) in outputs.iter().enumerate() {
        println!("  Output {}: {} - shape: {:?}", i, name, output.shape());
    }
    
    Ok(())
}
