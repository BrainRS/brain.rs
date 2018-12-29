use rand::prelude::*;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

pub type Signal = f64;

pub type InputData = Vec<Signal>;
pub type OutputData = Vec<Signal>;
pub type ErrorData = Vec<Signal>;

fn mse(errors: &OutputData) -> Signal {
    // mean squared error
    let mut sum = 0.0;
    for i in 0..errors.len() {
        sum += errors[i] * errors[i];
    }
    return sum / (errors.len() as Signal);
}

#[derive(Debug)]
pub struct TrainingDataSample {
    input: InputData,
    output: OutputData,
}
impl TrainingDataSample {
    pub fn new(input: InputData, output: OutputData) -> TrainingDataSample {
        TrainingDataSample { input, output }
    }
}

#[derive(Debug)]
pub struct TrainingData {
    samples: Vec<TrainingDataSample>,
    input_mapping: Option<HashMap<usize, &'static str>>,
    output_mapping: Option<HashMap<usize, &'static str>>,
}

impl TrainingData {
    pub fn new(samples: Vec<TrainingDataSample>) -> TrainingData {
        TrainingData {
            samples,
            input_mapping: None,
            output_mapping: None,
        }
    }
    pub fn new_with_mapping(
        samples: Vec<TrainingDataSample>,
        input_mapping: HashMap<usize, &'static str>,
        output_mapping: HashMap<usize, &'static str>,
    ) -> TrainingData {
        TrainingData {
            samples,
            input_mapping: Some(input_mapping),
            output_mapping: Some(output_mapping),
        }
    }
}

pub type InputObject = HashMap<&'static str, Signal>;
pub type OutputObject = HashMap<&'static str, Signal>;

#[derive(Debug)]
pub struct TrainingObjectSample {
    input: InputObject,
    output: OutputObject,
}
impl TrainingObjectSample {
    pub fn new(input: InputObject, output: OutputObject) -> TrainingObjectSample {
        TrainingObjectSample { input, output }
    }
}

#[derive(Debug)]
pub struct TrainingObject {
    samples: Vec<TrainingObjectSample>,
}

impl TrainingObject {
    pub fn new(samples: Vec<TrainingObjectSample>) -> TrainingObject {
        TrainingObject { samples }
    }
}

impl From<TrainingObject> for TrainingData {
    fn from(training_object: TrainingObject) -> TrainingData {
        let mut samples = vec![];
        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();
        let mut mapping_initialized = false;
        for sample in training_object.samples {
            let mut training_input = vec![];
            for (key, value) in sample.input {
                let index = training_input.len();
                if !mapping_initialized {
                    input_mapping.insert(index, key);
                }
                training_input.push(value);
            }
            let mut training_output = vec![];
            for (key, value) in sample.output {
                let index = training_output.len();
                if !mapping_initialized {
                    output_mapping.insert(index, key);
                }
                training_output.push(value);
            }
            if !mapping_initialized {
                mapping_initialized = true;
            }
            samples.push(TrainingDataSample::new(training_input, training_output));
        }
        TrainingData::new_with_mapping(samples, input_mapping, output_mapping)
    }
}

impl From<TrainingData> for TrainingObject {
    fn from(training_data: TrainingData) -> TrainingObject {
        let input_mapping = match training_data.input_mapping {
            Some(mapping) => mapping,
            None => panic!("This TrainingData doesn't contain input mapping information"),
        };
        let output_mapping = match training_data.output_mapping {
            Some(mapping) => mapping,
            None => panic!("This TrainingData doesn't contain output mapping information"),
        };
        let mut samples = vec![];
        for sample in training_data.samples {
            let mut training_input = HashMap::new();
            for (index, value) in sample.input.iter().enumerate() {
                let index_key = match input_mapping.get(&index) {
                    Some(index) => index,
                    None => panic!("This TrainingData doesn't contain input mapping information for index '{}'", index),
                };
                training_input.insert(index_key.clone(), *value);
            }
            let mut training_output = HashMap::new();
            for (index, value) in sample.output.iter().enumerate() {
                let index_key = match output_mapping.get(&index) {
                    Some(index) => index,
                    None => panic!("This TrainingData doesn't contain output mapping information for index '{}'", index),
                };
                training_output.insert(index_key.clone(), *value);
            }
            samples.push(TrainingObjectSample::new(training_input, training_output));
        }
        TrainingObject::new(samples)
    }
}

pub struct Neuron {
    bias: Signal,
    weights: Vec<Signal>,
    changes: Vec<Signal>,
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
            changes: (0..size).map(|_| 0.0).collect(),
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
        let mut neurons = vec![];
        for _i in 0..neuron_count {
            neurons.push(Neuron::new(input_count));
        }
        Layer { neurons }
    }
    fn get_outputs(&self) -> OutputData {
        let mut outputs = vec![];
        for neuron in &self.neurons {
            outputs.push(neuron.output);
        }
        outputs
    }
    fn get_deltas(&self) -> ErrorData {
        let mut deltas = vec![];
        for neuron in &self.neurons {
            deltas.push(neuron.delta);
        }
        deltas
    }
}

pub struct TrainingStatus {
    iterations: u32,
    error: Signal,
}

impl TrainingStatus {
    pub fn new() -> TrainingStatus {
        TrainingStatus {
            iterations: 0,
            error: 1.0, // Big enough number (larger than error_threshold) so we don't stop training prematurely
        }
    }
    pub fn get_iterations(&self) -> u32 {
        self.iterations
    }
    pub fn get_error(&self) -> Signal {
        self.error
    }
}

pub struct NeuralNetwork {
    options: NeuralNetworkOptions,
    layers: Vec<Layer>,
    error_check_interval: u32,
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
    pub output_layer_neuron_count: Option<usize>,
    pub hidden_layers: Option<Vec<usize>>,
    pub activation: NeuralActivation,
    pub iterations: u32,
    pub error_thresh: f64,
    pub log: bool,
    pub log_period: u32,
    pub learning_rate: f64,
    pub momentum: f64,
    pub callback: Option<Box<Fn(&TrainingStatus)>>,
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
            layers: vec![],
            error_check_interval: 1,
        };
        neural_network.initialize();
        neural_network
    }

    fn initialize(&mut self) {
        let default_hidden_layers = vec![];
        let hidden_layers = match &self.options.hidden_layers {
            Some(cnt) => cnt,
            None => &default_hidden_layers,
        };
        let input_layer_neuron_count = match self.options.input_layer_neuron_count {
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
        for hidden_layer_neuron_count in hidden_layers {
            let hidden_layer = Layer::new(*hidden_layer_neuron_count, input_count);
            input_count = *hidden_layer_neuron_count;
            self.layers.push(hidden_layer);
        }
        // Lastly add the output layer
        let output_layer = Layer::new(output_layer_neuron_count, input_count);
        self.layers.push(output_layer);
    }

    pub fn train(&mut self, training_data: &TrainingData) -> TrainingStatus {
        let mut status = TrainingStatus::new();
        let end_time = match self.options.timeout {
            Some(duration) => Some(SystemTime::now() + duration),
            None => None,
        };
        while self.training_tick(training_data, &mut status, end_time) {}
        status
    }

    fn calculate_training_error(&mut self, training_data: &TrainingData) -> Signal {
        let mut sum = 0.0;
        for training_sample in &training_data.samples {
            sum += self.train_sample(training_sample);
        }
        sum / (training_data.samples.len() as f64)
    }

    fn training_tick(
        &mut self,
        training_data: &TrainingData,
        status: &mut TrainingStatus,
        end_time: Option<SystemTime>,
    ) -> bool {
        if status.iterations >= self.options.iterations || status.error <= self.options.error_thresh
        {
            return false;
        }
        if let Some(end_time) = end_time {
            if SystemTime::now() >= end_time {
                return false;
            }
        }
        status.iterations += 1;
        if self.options.log && status.iterations % self.options.log_period == 0 {
            status.error = self.calculate_training_error(training_data);
            println!(
                "iterations: {}, training error: {}",
                status.iterations, status.error
            );
        } else {
            if status.iterations % self.error_check_interval == 0 {
                status.error = self.calculate_training_error(training_data);
            } else {
                self.train_samples(training_data);
            }
        }
        if status.iterations % self.options.callback_period == 0 {
            match &self.options.callback {
                Some(callback) => callback(status),
                None => (),
            }
        }
        return true;
    }

    fn train_samples(&mut self, training_data: &TrainingData) {
        for training_sample in &training_data.samples {
            self.train_sample(training_sample);
        }
    }

    fn train_sample(&mut self, training_sample: &TrainingDataSample) -> Signal {
        self.run_sample(&training_sample.input);
        self.calculate_deltas(&training_sample.output);
        self.adjust_weights();
        mse(&self.layers[self.layers.len() - 1].get_outputs())
    }

    fn run_sample(&mut self, input: &InputData) -> OutputData {
        match self.options.activation {
            NeuralActivation::Sigmoid => self
                .run_sample_with_activation(input, |sum: Signal| -> Signal {
                    1.0 / (1.0 + (-sum).exp())
                }),
            NeuralActivation::Relu => self.run_sample_with_activation(
                input,
                |sum: Signal| -> Signal {
                    if sum < 0.0 {
                        0.0
                    } else {
                        sum
                    }
                },
            ),
            NeuralActivation::LeakyRelu => {
                let alpha = self.options.leaky_relu_alpha;
                self.run_sample_with_activation(
                    input,
                    |sum: Signal| -> Signal {
                        if sum < 0.0 {
                            0.0
                        } else {
                            alpha * sum
                        }
                    },
                )
            }
            NeuralActivation::Tanh => {
                self.run_sample_with_activation(input, |sum: Signal| -> Signal { (-sum).tanh() })
            }
        }
    }

    fn run_sample_with_activation(
        &mut self,
        input: &InputData,
        activation_function: impl Fn(Signal) -> Signal,
    ) -> OutputData {
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
        let mut intermediate_input = vec![];
        intermediate_input.extend(input.iter().cloned());
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
            intermediate_input = vec![];
            intermediate_input.extend(layer.get_outputs().iter().cloned());
        }
        let mut output = vec![];
        {
            let output_layer = &self.layers[layer_count - 1];
            for i in 0..output_layer.neurons.len() {
                let neurons = &output_layer.neurons;
                let neuron = &neurons[i];
                output.push(neuron.output);
            }
        }
        output
    }

    fn calculate_deltas(&mut self, target: &OutputData) {
        match self.options.activation {
            NeuralActivation::Sigmoid => self.calculate_deltas_with_backward(
                target,
                |output: Signal, error: Signal| -> Signal { error * output * (1.0 - output) },
            ),
            NeuralActivation::Relu => self.calculate_deltas_with_backward(
                target,
                |output: Signal, error: Signal| -> Signal {
                    if output > 0.0 {
                        error
                    } else {
                        0.0
                    }
                },
            ),
            NeuralActivation::LeakyRelu => {
                let alpha = self.options.leaky_relu_alpha;
                self.calculate_deltas_with_backward(
                    target,
                    |output: Signal, error: Signal| -> Signal {
                        if output > 0.0 {
                            error
                        } else {
                            alpha * error
                        }
                    },
                )
            }
            NeuralActivation::Tanh => self.calculate_deltas_with_backward(
                target,
                |output: Signal, error: Signal| -> Signal { (1.0 - output * output) * error },
            ),
        }
    }

    fn calculate_deltas_with_backward(
        &mut self,
        target: &OutputData,
        backward_function: impl Fn(Signal, Signal) -> Signal,
    ) {
        for layer_index in (1..self.layers.len()).rev() {
            for neuron_index in 0..self.layers[layer_index].neurons.len() {
                let output = self.layers[layer_index].neurons[neuron_index].output;
                let mut error = 0.0;
                if layer_index == self.layers.len() - 1 {
                    error = target[neuron_index] - output;
                } else {
                    let deltas = self.layers[layer_index + 1].get_deltas();
                    for k in 0..deltas.len() {
                        error += deltas[k]
                            * self.layers[layer_index + 1].neurons[k].weights[neuron_index];
                    }
                }
                let neuron = &mut self.layers[layer_index].neurons[neuron_index];
                neuron.error = error;
                neuron.delta = backward_function(output, error);
            }
        }
    }

    fn adjust_weights(&mut self) {
        for layer_index in 1..self.layers.len() {
            let incoming = self.layers[layer_index - 1].get_outputs();
            for neuron_index in 0..self.layers[layer_index].neurons.len() {
                let neuron = &mut self.layers[layer_index].neurons[neuron_index];
                let delta = neuron.delta;
                for k in 0..incoming.len() {
                    let mut change = neuron.changes[k];
                    change = (self.options.learning_rate * delta * incoming[k])
                        + (self.options.momentum * change);
                    neuron.changes[k] = change;
                    neuron.weights[k] += change;
                }
                neuron.delta += self.options.learning_rate * delta;
            }
        }
    }

    pub fn run(&mut self, input_data: &InputData) -> OutputData {
        self.run_sample(input_data);
        self.layers.last().unwrap().get_outputs()
    }
}
