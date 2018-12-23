use std::time::Duration;

pub struct NeuralNetwork {
    _options: NeuralNetworkOptions,
}

pub struct NeuralNetworkOptions {
    // network options
    pub leaky_relu_alpha: f64,
    pub binary_thresh: f64,
    pub hidden_layers: Option<Vec<u32>>,  // array of ints for the sizes of the hidden layers in the network
    pub activation: String,              // supported activation types ['sigmoid', 'relu', 'leaky-relu', 'tanh']
    // training options
    pub iterations: u32,                 // the maximum times to iterate the training data
    pub error_thresh: f64,               // the acceptable error percentage from training data
    pub log: bool,                       // true to use console.log, when a function is supplied it is used
    pub log_period: u32,                 // iterations between logging out
    pub learning_rate: f64,              // multiply's against the input and the delta then adds to momentum
    pub momentum: f64,                   // multiply's against the specified "change" then adds to learning rate for change
    pub callback: Option<Box<Fn()>>,     // a periodic call back that can be triggered while training
    pub callback_period: u32,            // the number of iterations through the training data between callback calls
    pub timeout: Option<Duration>,       // the max number of milliseconds to train for
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
        return NeuralNetwork {
            _options: options,
        }
    }
}
