use std::time::Duration;

pub type Signal = f64;

pub type InputData = Vec<Signal>;

pub type OutputData = Vec<Signal>;

pub struct TrainingSample {
    input: InputData,
    output: OutputData,
}

impl TrainingSample {
    pub fn new(input: InputData, output: OutputData) -> TrainingSample {
        TrainingSample {
            input,
            output,
        }
    }
}

pub type TrainingData = Vec<TrainingSample>;

pub struct Neuron {
    weights: Vec<Signal>, // bias is on position 0
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

pub struct NeuralNetwork {
    options: NeuralNetworkOptions,
    layers: Option<Vec<Layer>>,
}

pub struct NeuralNetworkOptions {
    pub leaky_relu_alpha: f64,
    pub binary_thresh: f64,
    pub hidden_layers: Option<Vec<u32>>,
    pub activation: String,
    pub iterations: u32,
    pub error_thresh: f64,
    pub log: bool,
    pub log_period: u32,
    pub learning_rate: f64,
    pub momentum: f64,
    pub callback: Option<Box<Fn()>>,
    pub callback_period: u32,
    pub timeout: Option<Duration>,
    pub praxis: Option<String>,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}

impl Default for NeuralNetworkOptions {
    fn default() -> NeuralNetworkOptions {
        NeuralNetworkOptions {
            leaky_relu_alpha: 0.01,
            binary_thresh: 0.5,
            hidden_layers: None,
            activation: String::from("sigmoid"),
            iterations: 20000,
            error_thresh: 0.005,
            log: false,
            log_period: 10,
            learning_rate: 0.3,
            momentum: 0.1,
            callback: None,
            callback_period: 10,
            timeout: None,
            praxis: None,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

impl NeuralNetwork {
    pub fn new(options: NeuralNetworkOptions) -> NeuralNetwork {
        let mut neural_network = NeuralNetwork {
            options,
            layers: None, 
        };
        neural_network.initialize();
        neural_network
    }

    fn initialize(&mut self) {
        self.layers = Some(Vec::new());
    }

    pub fn train(&self, training_data: TrainingData) {

    }

    pub fn run(&self, input_data: InputData) -> OutputData {
        vec!(0.0)
    }
}
