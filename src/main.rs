extern crate brain;

use brain::neural_network::{NeuralNetwork, NeuralNetworkOptions, TrainingSample};

fn main() {
    let net_options = NeuralNetworkOptions {
        input_layer_neuron_count: Some(2),
        hidden_layers: Some(vec!(3)),
        output_layer_neuron_count: Some(1),
        iterations: 20000,
        log_period: 20001, // never log if this is bigger than iterations
        ..Default::default()
    };
    let mut net = NeuralNetwork::new(net_options);
    let training_data = vec!(
        TrainingSample::new(vec!(0.0, 0.0), vec!(0.0)),
        TrainingSample::new(vec!(0.0, 1.0), vec!(1.0)),
        TrainingSample::new(vec!(1.0, 0.0), vec!(1.0)),
        TrainingSample::new(vec!(1.0, 1.0), vec!(0.0)),
    );
    net.train(&training_data);
    println!("{:?}", net.run(&vec!(0.0, 0.0)));
    println!("{:?}", net.run(&vec!(0.0, 1.0)));
    println!("{:?}", net.run(&vec!(1.0, 0.0)));
    println!("{:?}", net.run(&vec!(1.0, 1.0)));
}
