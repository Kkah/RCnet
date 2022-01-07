from models import cnn, cnnlstm

def gen_model(args, device):
   
    assert args.model in [
        'cnn',
        'cnnlstm',
    ]
    
    if args.model == 'cnn':
        assert args.seq_len == 1, \
        "You must set --seq_len parameter as 1 to train with CNN;"
        
        print('\n## Importing CNN model')
        model = cnn.CNN(args)
        
        
    elif args.model == 'cnnlstm':
        print('\n## Importing CNNLSTM model')
        model = cnnlstm.CNNLSTM(args)
        
    return model.to(device)
        