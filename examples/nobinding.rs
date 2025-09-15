use ort::{
    session::Session,
    execution_providers::TensorRTExecutionProvider,
    value::Tensor,
    Result,
};
use ndarray::Array4;
use std::time::Instant;

fn main() -> Result<()> {
    // Init environment with TensorRT EP
    let _env = ort::init()
        .with_name("tensorrt_no_iobinding")
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

    println!("Session ready with TensorRT (no IOBinding)!");

    // -------- CPU tensor allocation --------
    let input_data: Array4<f32> = Array4::zeros((1, 3, 512, 512));
    let input_tensor = Tensor::from_array(input_data)?;

    // -------- Timer for inference --------
    let start = Instant::now();
    let outputs = session.run(ort::inputs![input_tensor])?;
    let duration = start.elapsed();

    println!("Inference finished in {:?}", duration);

    Ok(())
}
