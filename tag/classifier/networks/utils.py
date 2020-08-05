try:
    # print("torch.hub.load_state_dict_from_url")
    from torch.hub import load_state_dict_from_url
except ImportError:
    # print("torch.utils.model_zoo.load_state_dict_from_url")
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
