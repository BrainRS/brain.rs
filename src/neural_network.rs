use rand::prelude::*;
use std::time::Duration;

fn clone_vector<T> (oldVector: Vec<T>) -> Vec<T> {
    let mut newVector = Vec::<T>::new();
    for elem in oldVector {
        newVector.push(elem);
    }
    newVector
}

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
    bias: Signal,
    weights: Vec<Signal>,
    output: Signal,
    error: Signal,
}

impl Neuron {
    fn new(size: usize) -> Neuron {
        let mut rng = rand::thread_rng();

        Neuron {
            bias: rng.gen(),
            weights: (0..size).map(|_| rng.gen()).collect(),
            output: 0.0,
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
}

pub struct NeuralNetwork {
    options: NeuralNetworkOptions,
    layers: Vec<Layer>,
}

pub struct NeuralNetworkOptions {
    pub leaky_relu_alpha: f64,
    pub binary_thresh: f64,
    pub input_layer_neuron_count: Option<usize>,
    pub hidden_layers_neuron_count: Option<usize>,
    pub output_layer_neuron_count: Option<usize>,
    pub hidden_layers: Option<usize>,
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
            input_layer_neuron_count: None,
            hidden_layers_neuron_count: None,
            output_layer_neuron_count: None,
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

    fn train_sample(&mut self, training_sample: TrainingSample) -> Signal {
        self.run_sample(training_sample.input);
        self.calculate_deltas(training_sample.output);
        self.adjust_weights();
        0.0
    }

    fn run_sample(&mut self, input: InputData) -> OutputData {
        match &self.options.activation[..] {
            "sigmoid" => self.run_sample_with_activation(input, |sum: Signal| -> Signal {
                1.0 / (1.0 + (-sum).exp())
            }),
            "relu" => self.run_sample_with_activation(input, |sum: Signal| -> Signal {
                if sum < 0.0 {
                    0.0
                } else {
                    sum
                }
            }),
            "leaky-relu" => {
                let alpha = self.options.leaky_relu_alpha;
                self.run_sample_with_activation(input, |sum: Signal| -> Signal {
                    if sum < 0.0 {
                        0.0
                    } else {
                        alpha * sum
                    }
                })
            },
            "tanh" => self.run_sample_with_activation(input, |sum: Signal| -> Signal {
                (-sum).tanh()
            }),
            _ => panic!("run_sample called with unknown activation '{}'", self.options.activation),
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
        println!("intermediate_input: {:?}", intermediate_input);
        let mut first_layer_skipped = false;
        for layer in &mut self.layers {
            if !first_layer_skipped {
                first_layer_skipped = true;
                continue;
            }
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
            let output_layer = &mut self.layers[layer_count-1];
            for i in 0..output_layer.neurons.len() {
                let neurons = &mut output_layer.neurons;
                let neuron = &mut neurons[i];
                output.push(neuron.output);
            }
        }
        output
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

    pub fn run(&mut self, input_data: InputData) -> OutputData {
        self.run_sample(input_data);
        vec!(0.0)
    }
}
