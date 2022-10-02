"""
For some complex methods with multiple network modules.
"""

from models.BYOL import BYOL
from models.PCRL import PCRLModel3d, PCRLModel3d_AB

models_dict= {
    'BYOL': BYOL,
    #'Pixel_BYOL': PixelBYOL,
    'PCRL_Model': PCRLModel3d,

}

def get_models(args, network):
    model_name = args.model
    if model_name == "BYOL":
        model = BYOL(network, hidden_dim=512, pred_dim=256, m=0.996)

    elif model_name == 'PCRL_Model':
        model = PCRLModel3d(encoder=network[0],
                            encoder_ema=network[1],
                            decoder=network[2])

    else:
        raise  NotImplementedError("The model does not exists!")
    return model
