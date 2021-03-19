def network_factory(name: str):
    if name == 'cin':
        from networks.cin import Cin as Network
    elif name == 'dcn':
        from networks.dcn import Dcn as Network
    elif name == 'deepfm':
        from networks.deepfm import DeepFM as Network
    elif name == "dnn":
        from networks.dnn import Dnn as Network
    elif name == "enfm":
        from networks.enfm import NeuralFM as Network
    elif name == "lr":
        from networks.lr import LR as Network
    elif name == 'mvm':
        from networks.mvm import Mvm as Network
    elif name == 'nfm':
        from networks.nfm import Nfm as Network
    elif name == 'fm':
        from networks.fm import Fm as Network
    elif name == 'deepfm_bias_gate_cross':
        from networks.deepfm_bias_gate_cross import DeepFM_bias_gate_cross as Network
    else:
        raise ValueError('{} network is not implement yet'.format(name))
    return Network
