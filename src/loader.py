import os

import torch

from models import LinkSeg, FrameEncoder
from collections import OrderedDict

def load_model(args):
    r"""Load a trained model from a checkpoint file.
    Args:
        checkpoint (str): path to the checkpoint or name of the checkpoint file (if using a provided checkpoint)
    Returns:
        LinkSeg: instance of LinkSeg model
    """

    print('CHECKPOINT =', args.model_name)
    if os.path.exists(args.model_name):  # handle user-provided checkpoints
        model_path = args.model_name
    else:
        model_path = os.path.join(os.path.dirname(__file__), "weights", args.model_name + ".pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"You passed an invalid checkpoint file: {args.model_name}.")
    
    # load checkpoint
    encoder = FrameEncoder(
                n_mels=args.n_mels,
                conv_ndim=args.conv_ndim,
                sample_rate=args.sample_rate,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_embedding=args.n_embedding,
                f_min=args.f_min,
                f_max=args.f_max,
                dropout=args.dropout,
                hidden_dim=args.hidden_dim,
                attention_ndim=args.attention_ndim,
                attention_nlayers=args.attention_nlayers,
                attention_nheads=args.attention_nheads,
    )

    model = LinkSeg(
                encoder=encoder, 
                nb_ssm_classes=args.nb_ssm_classes, 
                nb_section_labels=args.nb_section_labels, 
                hidden_size=args.hidden_size, 
                output_channels=args.output_channels,
                dropout_gnn=args.dropout_gnn,
                dropout_cnn=args.dropout_cnn,
                dropout_egat=args.dropout_egat,
                max_len=args.max_len,
    )

    print('Model path =', model_path)
    state_dict = torch.load(model_path, weights_only=True)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():

        name = k.split('network.')[-1]
        #if ('front_end.frontend' in name) and ('front_end.frontend.fc' not in name):
        #if 'front_end.pos_embedding' not in name:
        new_state_dict[name] = v

    
    model.load_state_dict(new_state_dict, strict=True) 
    

    # instantiate LinkSeg encoder
    #model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model