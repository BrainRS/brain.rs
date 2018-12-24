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
    // neurons: Vec<Neuron>,
}

pub struct NeuralNetwork {
    options: NeuralNetworkOptions,
    layers: Vec<Layer>,
    outputs: Vec<OutputData>,
}

pub struct NeuralNetworkOptions {
    pub leaky_relu_alpha: f64,
    pub binary_thresh: f64,
    pub hidden_layers: Option<u32>,
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
        let layers = NeuralNetwork::build_layers(&options);
        NeuralNetwork {
            options,
            layers,
        }
    }

    fn build_layers(options: &NeuralNetworkOptions) -> Vec<Layer> {
        let hidden_layers_count = match options.hidden_layers {
            Some(cnt) => cnt,
            None => 3,
        };
        let mut layers = Vec::new();
        // First add the input layer
        let input_layer = Layer {};
        layers.push(input_layer);
        // Add the hidden layers
        for i in 0..hidden_layers_count {
            let hidden_layer = Layer {};
            layers.push(hidden_layer);
        }
        // Lastly add the output layer
        let output_layer = Layer {};
        layers.push(output_layer);
        layers
    }

    pub fn train(&self, training_data: TrainingData) {
        for training_sample in training_data {
            self.train_sample(training_sample);
        }
    }

    fn train_sample(&self, training_sample: TrainingSample) -> Signal {
        self.run_sample(training_sample.input);
        self.calculate_deltas(training_sample.output);
        self.adjust_weights();
        0.0
    }

    fn run_sample(&self, input: InputData) -> OutputData {
        match &self.options.activation[..] {
            "sigmoid" => self.run_sample_sigmoid(input),
            "relu" => self.run_sample_relu(input),
            "leaky-relu" => self.run_sample_leaky_relu(input),
            "tanh" => self.run_sample_tanh(input),
            _ => panic!("run_sample called with unknown activation '{}'", self.options.activation),
        }
    }

    fn run_sample_with_activation(&mut self, input: InputData, activationFunction: impl FnMut(Signal)->Signal) -> OutputData {
        self.outputs[0] = input;  // set output state of input layer

        let mut output = vec!();
        for layer in 1..self.outputLayer {
            for node in 0..self.sizes[layer] {
                let weights = self.weights[layer][node];

                let sum = self.biases[layer][node];
                for k in 0..weights.length {
                    sum += weights[k] * input[k];
                }
                //sigmoid
                self.outputs[layer][node] = activationFunction(sum);
            }
            input = self.outputs[layer];
            output = input;
        }
        output
    }

    fn run_sample_sigmoid(&mut self, input: InputData) -> OutputData {
        self.run_sample_with_activation(input, |sum: Signal| -> Signal {
            1.0 / (1.0 + (-sum).exp())
        })
    }

    fn run_sample_relu(&self, input: InputData) -> OutputData {
        self.run_sample_with_activation(input, |sum: Signal| -> Signal {
            if sum < 0.0 {
                0.0
            } else {
                sum
            }
        })
    }

    fn run_sample_leaky_relu(&self, input: InputData) -> OutputData {
        let alpha = self.options.leaky_relu_alpha;
        self.run_sample_with_activation(input, |sum: Signal| -> Signal {
            if sum < 0.0 {
                0.0
            } else {
                alpha * sum
            }
        })
    }

    fn run_sample_tanh(&self, input: InputData) -> OutputData {
        self.run_sample_with_activation(input, |sum: Signal| -> Signal {
            (-sum).tanh()
        })
    }

    fn calculate_deltas(&self, output: OutputData) {
        match &self.options.activation[..] {
            "sigmoid" => self.calculate_deltas_sigmoid(output),
            "relu" => self.calculate_deltas_relu(output),
            "leaky-relu" => self.calculate_deltas_leaky_relu(output),
            "tanh" => self.calculate_deltas_tanh(output),
            _ => panic!("calculate_deltas called with unknown activation '{}'", self.options.activation),
        }
    }

    fn calculate_deltas_sigmoid(&self, output: OutputData) {
    }

    fn calculate_deltas_relu(&self, output: OutputData) {
    }

    fn calculate_deltas_leaky_relu(&self, output: OutputData) {
    }

    fn calculate_deltas_tanh(&self, output: OutputData) {
    }

    fn adjust_weights(&self) {
        // for (let layer = 1; layer <= this.outputLayer; layer++) {
        //     let incoming = this.outputs[layer - 1];

        //     for (let node = 0; node < this.sizes[layer]; node++) {
        //         let delta = this.deltas[layer][node];

        //         for (let k = 0; k < incoming.length; k++) {
        //             let change = this.changes[layer][node][k];

        //             change = (this.trainOpts.learningRate * delta * incoming[k])
        //             + (this.trainOpts.momentum * change);

        //             this.changes[layer][node][k] = change;
        //             this.weights[layer][node][k] += change;
        //         }
        //         this.biases[layer][node] += this.trainOpts.learningRate * delta;
        //     }
        // }
    }

    pub fn run(&self, input_data: InputData) -> OutputData {
        self.run_sample(input_data);
        vec!(0.0)
    }
}
