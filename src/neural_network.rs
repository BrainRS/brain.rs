use rand::prelude::*;
use std::time::Duration;

fn clone_vector<T> (source_vector: Vec<T>) -> Vec<T> {
    let mut copied_vector = Vec::<T>::new();
    for elem in source_vector {
        copied_vector.push(elem);
    }
    copied_vector
}

pub type Signal = f64;
pub type InputData = Vec<Signal>;
pub type OutputData = Vec<Signal>;
pub type ErrorData = Vec<Signal>;
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
    bias: Signal,
    weights: Vec<Signal>,
    output: Signal,
    delta: Signal,
    error: Signal,
}

impl Neuron {
    fn new(size: usize) -> Neuron {
        let mut rng = rand::thread_rng();

        Neuron {
            bias: rng.gen(),
            weights: (0..size).map(|_| rng.gen()).collect(),
            output: 0.0,
            delta: 0.0,
            error: 0.0,
        }
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(neuron_count: usize, input_count: usize) -> Layer {
        let mut neurons = vec!();
        for _i in 0..neuron_count {
            neurons.push(Neuron::new(input_count));
        }
        Layer {
            neurons,
        }
    }
    fn get_outputs(&self) -> OutputData {
        let mut outputs = vec!();
        for neuron in &self.neurons {
            outputs.push(neuron.output);
        }
        outputs
    }
    fn get_deltas(&self) -> ErrorData {
        let mut deltas = vec!();
        for neuron in &self.neurons {
            deltas.push(neuron.delta);
        }
        deltas
    }
}

pub struct NeuralNetwork {
    options: NeuralNetworkOptions,
    layers: Vec<Layer>,
}

pub enum NeuralActivation {
    Sigmoid,
    Relu,
    LeakyRelu,
    Tanh,
}

pub struct NeuralNetworkOptions {
    pub leaky_relu_alpha: f64,
    pub binary_thresh: f64,
    pub input_layer_neuron_count: Option<usize>,
    pub hidden_layers_neuron_count: Option<usize>,
    pub output_layer_neuron_count: Option<usize>,
    pub hidden_layers: Option<usize>,
    pub activation: NeuralActivation,
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
            input_layer_neuron_count: None,
            hidden_layers_neuron_count: None,
            output_layer_neuron_count: None,
            hidden_layers: None,
            activation: NeuralActivation::Sigmoid,
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
            layers: vec!(),
        };
        neural_network.initialize();
        neural_network
    }

    fn initialize(&mut self) {
        let hidden_layers = match self.options.hidden_layers {
            Some(cnt) => cnt,
            None => 0,
        };
        let input_layer_neuron_count = match self.options.input_layer_neuron_count {
            Some(cnt) => cnt,
            None => 0,
        };
        let hidden_layers_neuron_count = match self.options.hidden_layers_neuron_count {
            Some(cnt) => cnt,
            None => 0,
        };
        let output_layer_neuron_count = match self.options.output_layer_neuron_count {
            Some(cnt) => cnt,
            None => 0,
        };
        // First add the input layer
        let input_layer = Layer::new(input_layer_neuron_count, 0);
        self.layers.push(input_layer);
        // Initialize input_count
        let mut input_count = input_layer_neuron_count;
        // Add the hidden layers
        for _i in 0..hidden_layers {
            let hidden_layer = Layer::new(hidden_layers_neuron_count, input_count);
            input_count = hidden_layers_neuron_count;
            self.layers.push(hidden_layer);
        }
        // Lastly add the output layer
        let output_layer = Layer::new(output_layer_neuron_count, input_count);
        self.layers.push(output_layer);
    }

    pub fn train(&mut self, training_data: TrainingData) {
        for training_sample in training_data {
            self.train_sample(training_sample);
        }
    }

    fn train_sample(&mut self, training_sample: TrainingSample) -> OutputData {
        self.run_sample(training_sample.input);
        self.calculate_deltas(training_sample.output);
        self.adjust_weights();
        self.layers[self.layers.len()-1].get_outputs()
    }

    fn run_sample(&mut self, input: InputData) -> OutputData {
        match self.options.activation {
            NeuralActivation::Sigmoid => self.run_sample_with_activation(input, |sum: Signal| -> Signal {
                1.0 / (1.0 + (-sum).exp())
            }),
            NeuralActivation::Relu => self.run_sample_with_activation(input, |sum: Signal| -> Signal {
                if sum < 0.0 {
                    0.0
                } else {
                    sum
                }
            }),
            NeuralActivation::LeakyRelu => {
                let alpha = self.options.leaky_relu_alpha;
                self.run_sample_with_activation(input, |sum: Signal| -> Signal {
                    if sum < 0.0 {
                        0.0
                    } else {
                        alpha * sum
                    }
                })
            },
            NeuralActivation::Tanh => self.run_sample_with_activation(input, |sum: Signal| -> Signal {
                (-sum).tanh()
            }),
        }
    }

    fn run_sample_with_activation(&mut self, input: InputData, activation_function: impl Fn(Signal)->Signal) -> OutputData {
        let layer_count = self.layers.len();
        {
            let input_layer = &mut self.layers[0];
            if input_layer.neurons.len() != input.len() {
                panic!("Input neurons and input values contain different number of elements ({} vs {})", input_layer.neurons.len(), input.len());
            }
            for i in 0..input_layer.neurons.len() {
                let neurons = &mut input_layer.neurons;
                let neuron = &mut neurons[i];
                neuron.output = input[i];
            }
        }
        let mut intermediate_input = clone_vector(input);
        for layer_index in 1..self.layers.len() {
            let layer = &mut self.layers[layer_index];
            let neurons = &mut layer.neurons;
            for neuron in neurons {
                let weights = &neuron.weights;
                let mut sum = neuron.bias;
                for k in 0..weights.len() {
                    sum += weights[k] * intermediate_input[k];
                }
                neuron.output = activation_function(sum);
            }
            intermediate_input = clone_vector(layer.get_outputs());
        }
        let mut output = vec!();
        {
            let output_layer = &self.layers[layer_count-1];
            for i in 0..output_layer.neurons.len() {
                let neurons = &output_layer.neurons;
                let neuron = &neurons[i];
                output.push(neuron.output);
            }
        }
        output
    }

    fn calculate_deltas(&mut self, target: OutputData) {
        match self.options.activation {
            NeuralActivation::Sigmoid => self.calculate_deltas_with_backward(target, |output: Signal, error: Signal| -> Signal {
                error * output * (1.0 - output)
            }),
            NeuralActivation::Relu => self.calculate_deltas_with_backward(target, |output: Signal, error: Signal| -> Signal {
                if output > 0.0 {
                    error
                } else {
                    0.0
                }
            }),
            NeuralActivation::LeakyRelu => {
                let alpha = self.options.leaky_relu_alpha;
                self.calculate_deltas_with_backward(target, |output: Signal, error: Signal| -> Signal {
                    if output > 0.0 {
                        error
                    } else {
                        alpha * error
                    }
                })
            },
            NeuralActivation::Tanh => self.calculate_deltas_with_backward(target, |output: Signal, error: Signal| -> Signal {
                (1.0 - output * output) * error
            }),
        }
    }

    fn calculate_deltas_with_backward(&mut self, target: OutputData, backward_function: impl Fn(Signal, Signal)->Signal) {
        for layer_index in (1..self.layers.len()).rev() {
            for neuron_index in 0..self.layers[layer_index].neurons.len() {
                let output = self.layers[layer_index].neurons[neuron_index].output;
                let mut error = 0.0;
                if layer_index == self.layers.len() - 1 {
                    error = target[neuron_index] - output;
                } else {
                    let deltas = self.layers[layer_index+1].get_deltas();
                    for k in 0..deltas.len() {
                        error += deltas[k] * self.layers[layer_index+1].neurons[k].weights[neuron_index];
                    }
                }
                let neuron = &mut self.layers[layer_index].neurons[neuron_index];
                neuron.error = error;
                neuron.delta = backward_function(output, error);
                println!("Neuron: error={} delta={}", neuron.error, neuron.delta);
            }
        }
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

    pub fn run(&mut self, input_data: InputData) -> OutputData {
        self.run_sample(input_data);
        vec!(0.0)
    }
}
