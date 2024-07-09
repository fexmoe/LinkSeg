import os

import torch


from model import LinkSeg, Backbone
from collections import OrderedDict

def load_model(checkpoint: str, config: dict) -> LinkSeg:
    r"""Load a trained model from a checkpoint file.
    Args:
        checkpoint (str): path to the checkpoint or name of the checkpoint file (if using a provided checkpoint)
    Returns:
        LinkSeg: instance of LinkSeg model
    """
    if os.path.exists(checkpoint):  # handle user-provided checkpoints
        model_path = checkpoint
    else:
        model_path = os.path.join(os.path.dirname(__file__), "weights", checkpoint + ".pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"You passed an invalid checkpoint file: {checkpoint}.")
    
    # load checkpoint
    encoder = Backbone(config)

    # instantiate main LinkSeg module and load its weights
    model = LinkSeg(config=config, encoder=encoder)
    print('Model path =', model_path)
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():

        name = k.split('network.')[-1]
        #if ('front_end.frontend' in name) and ('front_end.frontend.fc' not in name):
        #if 'front_end.pos_embedding' not in name:
        new_state_dict[name] = v

    
    model.load_state_dict(new_state_dict, strict=False) 
    

    # instantiate LinkSeg encoder
    #model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model