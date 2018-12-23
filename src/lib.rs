pub mod cross_validate;
pub mod likely;
pub mod lookup;
pub mod neural_network;
pub mod neural_network_gpu;
pub mod train_stream;
pub mod recurrent {
    mod rnn;
    mod lstm;
    mod gru;
    mod rnn_time_step;
    mod lstm_time_step;
    mod gru_time_step;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
