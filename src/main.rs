extern crate brain;

use brain::neural_network::{NeuralNetwork, NeuralNetworkOptions, TrainingSample};

fn main() {
    let net_options = NeuralNetworkOptions {
        hidden_layers: Some(vec!(3)),
        ..Default::default()
    };
    let net = NeuralNetwork::new(net_options);
    let training_data = vec!(
        TrainingSample::new(vec!(0.0, 0.0), vec!(0.0)),
        TrainingSample::new(vec!(0.0, 1.0), vec!(1.0)),
        TrainingSample::new(vec!(1.0, 0.0), vec!(1.0)),
        TrainingSample::new(vec!(1.0, 1.0), vec!(0.0)),
    );
    net.train(training_data);
    println!("{:?}", net.run(vec!(0.0, 0.0)));
}
