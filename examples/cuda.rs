use ort::{
    session::Session,
    execution_providers::CUDAExecutionProvider,
    value::Tensor,
    Result,
};
use ndarray::Array4;
use std::time::Instant;

fn main() -> Result<()> {
    // Init environment with CUDA EP
    let _env = ort::init()
        .with_name("ort_cuda_env")
        .with_telemetry(false)
        .with_execution_providers([
            CUDAExecutionProvider::default()
                .with_device_id(0)   // use GPU 0
                .build(),
        ])
        .commit();

    let model_path = "./YOLOv5.onnx";
    let mut session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?   // not really important for GPU, but safe
        .commit_from_file(model_path)?;

    println!("Model inputs:");
    for (i, input) in session.inputs.iter().enumerate() {
        println!("  Input {}: {} ({:?})", i, input.name, input.input_type);
    }

    println!("Model outputs:");
    for (i, output) in session.outputs.iter().enumerate() {
        println!("  Output {}: {} ({:?})", i, output.name, output.output_type);
    }

    let input_shape = (1, 3, 512, 512);
    let input_data: Array4<f32> = Array4::zeros(input_shape);

    let input_tensor = Tensor::from_array(input_data)?;
    let start = Instant::now();
    session.run(ort::inputs![input_tensor])?;
    let duration = start.elapsed();

    println!("Inference done in {:?}", duration);
    Ok(())
}
