use std::time::Duration;

#[derive(Debug)]
pub struct NeuralNetwork {
}

pub struct NeuralNetworkOptions {
    pub iterations: u32,              // the maximum times to iterate the training data
    pub error_thresh: f64,             // the acceptable error percentage from training data
    pub log: bool,                    // true to use console.log, when a function is supplied it is used
    pub log_period: u32,               // iterations between logging out
    pub learning_rate: f64,            // multiply's against the input and the delta then adds to momentum
    pub momentum: f64,                // multiply's against the specified "change" then adds to learning rate for change
    pub callback: Option<Box<Fn()>>,  // a periodic call back that can be triggered while training
    pub callback_period: u32,          // the number of iterations through the training data between callback calls
    pub timeout: Option<Duration>,    // the max number of milliseconds to train for
    pub praxis: Option<String>,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}

impl Default for NeuralNetworkOptions {
    fn default() -> NeuralNetworkOptions {
        NeuralNetworkOptions {
            iterations: 20000,    // the maximum times to iterate the training data
            error_thresh: 0.005,   // the acceptable error percentage from training data
            log: false,           // true to use console.log, when a function is supplied it is used
            log_period: 10,        // iterations between logging out
            learning_rate: 0.3,    // multiply's against the input and the delta then adds to momentum
            momentum: 0.1,        // multiply's against the specified "change" then adds to learning rate for change
            callback: None,       // a periodic call back that can be triggered while training
            callback_period: 10,   // the number of iterations through the training data between callback calls
            timeout: None,        // the max number of milliseconds to train for
            praxis: None,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}
impl NeuralNetwork {
    pub fn new(_options: NeuralNetworkOptions) -> NeuralNetwork {
        return NeuralNetwork {}
    }
}
