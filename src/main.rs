extern crate brain;

use brain::neural_network::{NeuralNetwork, NeuralNetworkOptions, TrainingSample};

fn main() {
    let net_options = NeuralNetworkOptions {
        input_layer_neuron_count: Some(2),
        hidden_layers: Some(5),
        hidden_layers_neuron_count: Some(4),
        output_layer_neuron_count: Some(1),
        ..Default::default()
    };
    let mut net = NeuralNetwork::new(net_options);
    let training_data = vec!(
        TrainingSample::new(vec!(0.0, 0.0), vec!(0.0)),
        TrainingSample::new(vec!(0.0, 1.0), vec!(1.0)),
        TrainingSample::new(vec!(1.0, 0.0), vec!(1.0)),
        TrainingSample::new(vec!(1.0, 1.0), vec!(0.0)),
    );
    //net.train(training_data);
    println!("{:?}", net.run(vec!(0.0, 0.0)));
}
