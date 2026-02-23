
import torch
from hyperpyyaml import load_hyperpyyaml
from safetensors.torch import load_file as safe_load_file

def test_tta_encoder():
    # wget https://huggingface.co/AudenAI/auden-encoder-tta-m10/resolve/main/model.safetensors
    config_file = "./tests/yaml/test_tta_encoder.yaml"
    with open(config_file) as fin:
        cfg = load_hyperpyyaml(fin)

    model = cfg["speech_encoder"]
    print(model)

    weight_path = './tests/model.safetensors'
    map_location = "cpu"

    device_arg = (
        str(map_location)
        if isinstance(map_location, torch.device)
        else map_location
    )
    state_obj = safe_load_file(weight_path, device=device_arg)
    state_dict = (
        state_obj["state_dict"]
        if isinstance(state_obj, dict) and "state_dict" in state_obj
        else state_obj
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    x = torch.rand([2, 1836, 80])
    x_lens = torch.tensor([1836, 382])
    y, y_lens = model(x, x_lens)
    
    print(y.shape, y_lens)


if __name__ == "__main__":
    test_tta_encoder()