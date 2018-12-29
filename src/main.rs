extern crate brain;
#[macro_use] extern crate maplit;

use brain::neural_network::*;

fn main() {
    let net_options = NeuralNetworkOptions {
        input_layer_neuron_count: Some(2),
        hidden_layers: Some(vec!(3)),
        output_layer_neuron_count: Some(1),
        log: true,
        ..Default::default()
    };
    let mut net = NeuralNetwork::new(net_options);
    let training_object = TrainingObject::new(vec!(
        TrainingObjectSample::new(hashmap!("high" => 0.0, "low" => 0.0), hashmap!("muscle" => 0.0)),
        TrainingObjectSample::new(hashmap!("high" => 0.0, "low" => 1.0), hashmap!("muscle" => 1.0)),
        TrainingObjectSample::new(hashmap!("high" => 1.0, "low" => 0.0), hashmap!("muscle" => 1.0)),
        TrainingObjectSample::new(hashmap!("high" => 1.0, "low" => 1.0), hashmap!("muscle" => 0.0)),
    ));
    let training_data = TrainingData::from(training_object);
    println!("{:?}", training_data);
    net.train(&training_data);
    println!("{:?}", net.run(&vec!(0.0, 0.0)));
    println!("{:?}", net.run(&vec!(0.0, 1.0)));
    println!("{:?}", net.run(&vec!(1.0, 0.0)));
    println!("{:?}", net.run(&vec!(1.0, 1.0)));
}
