import argparse

def parse_enable_gpu(enable_gpu):
    if enable_gpu == 1:
        return True
    elif enable_gpu == 0:
        return False
    raise ValueError("Enable GPU Should Either 0 or 1")

def get_train_cli_args():
    parser = argparse.ArgumentParser(description='')
    
    # Add arguments
    parser.add_argument('data_dir', type=str, help='Path to the data directory')
    parser.add_argument('--arch_id', type=int, default=0, help='Architecture ID to be used during training')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate to be used during training')
    parser.add_argument('--hidden_units', type=int, default=256, help='Number of hidden units to be used during training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to be used during training')
    parser.add_argument('--enable_gpu', type=int, default=1, help='Whether to use GPU during training. Valid inputs are 0 or 1')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    data_dir = args.data_dir
    arch_id = args.arch_id
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs

    # Get gpu enabled
    enable_gpu = parse_enable_gpu(args.enable_gpu)
        
    return data_dir, arch_id, learning_rate, hidden_units, epochs, enable_gpu

def get_predict_cli_args():
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('image_path', type=str, help='Path to the image to predict')
    parser.add_argument('checkpoint_path', type=str, help='Path to the model checkpoint')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the classifier names')
    parser.add_argument('--top_k', type=int, default=-1, help='Amount of most likely classes to display')
    parser.add_argument('--enable_gpu', type=int, default=1, help='Whether to use GPU during training. Valid inputs are 0 or 1')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    image_path = args.image_path
    checkpoint_path = args.checkpoint_path
    category_names = args.category_names
    top_k = args.top_k
    
    # Get gpu enabled
    enable_gpu = parse_enable_gpu(args.enable_gpu)
    
    return image_path, checkpoint_path, category_names, top_k, enable_gpu
  
    