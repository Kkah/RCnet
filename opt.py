import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Robotic Colonosocpy models.')

    parser.add_argument('--model', type=str,
                        help='select model in (cnn | cnnlstm)', default = 'cnn')
    parser.add_argument('--data_path', type=str,
                        help='path to dataset', default = 'C:/Users/217/RC/dataset')
    parser.add_argument('--save_model_path', type=str,
                        help='path to saving model', default = './saved_model')
    parser.add_argument('--hidden1', type=int,
                        help='number of nodes in hidden layer 1', default = 256)
    parser.add_argument('--hidden2', type=int,
                        help='number of nodes in hidden layer 2', default = 128)
    parser.add_argument('--output_size', type=int,
                        help='number of nodes in hidden layer 2', default = 2)
    parser.add_argument('--CNN_embed_dim', type=int,
                        help='dimension of CNN embedding vector', default = 512)
    parser.add_argument('--h_RNN', type=int,
                        help='number of nodes in LSTM', default = 256)
    parser.add_argument('--h_FC_dim', type=int,
                        help='dimension of fc layer in RNN Decoder', default = 128)
    parser.add_argument('--num_layers', type=int,
                        help='number of layers of LSTM', default = 1)
    parser.add_argument('--dropout', type=float,
                        help='dropout rate 0~1', default= 0)
    parser.add_argument('--epochs', type=int,
                        help='number of epoches to iterate', default= 20)
    parser.add_argument('--lr', type=float,
                        help='learning rate', default= 1e-3)
    parser.add_argument('--batch_size', type=int,
                        help='number of batch_size', default= 256)
    parser.add_argument('--seq_len', type=int,
                        help='number of sequence length, if it is only cnn set to 1', default = 1)
    parser.add_argument('--target_len', type=int,
                        help='number of target length, if it is only cnn set to 1', default = 1)
    parser.add_argument('--start_skip_frame', type=int,
                        help='number of frames to skip from start', default = 20)
    parser.add_argument('--end_skip_frame', type=int,
                        help='number of frames to skip from end', default = 20)
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    
    args = parser.parse_args()
    
    return args