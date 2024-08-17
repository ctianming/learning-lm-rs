use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        // let get_tensor: impl Fn(&str) -> Tensor<f32> = |name: &str| {
        // ...
        // };

        // LLamaParams {
        //     embedding_table: get_tensor(...),
        //     ...
        // }
        // 定义 get_tensor 函数
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor_view = safetensor
                .tensor(name)
                .expect(&format!("Tensor {} not found", name));
            // 将 TensorView 转换为 Tensor
            let data = tensor_view.data();
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();

            Tensor::new(f32_data, &tensor_view.shape().to_vec())
        };

        LLamaParams {
            wv: (0..config.hidden_size)
                .map(|i| get_tensor(&format!("wv_{}", i)))
                .collect(),
            wo: (0..config.hidden_size)
                .map(|i| get_tensor(&format!("wo_{}", i)))
                .collect(),
            rms_ffn_w: (0..config.hidden_size)
                .map(|i| get_tensor(&format!("rms_ffn_w_{}", i)))
                .collect(),
            w_up: (0..config.hidden_size)
                .map(|i| get_tensor(&format!("w_up_{}", i)))
                .collect(),
            w_gate: (0..config.hidden_size)
                .map(|i| get_tensor(&format!("w_gate_{}", i)))
                .collect(),
            w_down: (0..config.hidden_size)
                .map(|i| get_tensor(&format!("w_down_{}", i)))
                .collect(),
            rms_out_w: get_tensor("rms_out_w"),
            lm_head: get_tensor("lm_head"),
            // 初始化缺少的字段
            embedding_table: get_tensor("embedding_table"),
            rms_att_w: (0..config.hidden_size)
                .map(|i| get_tensor(&format!("rms_att_w_{}", i)))
                .collect(),
            wk: (0..config.hidden_size)
                .map(|i| get_tensor(&format!("wk_{}", i)))
                .collect(),
            wq: (0..config.hidden_size)
                .map(|i| get_tensor(&format!("wq_{}", i)))
                .collect(),
        }
    }
}
