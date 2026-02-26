from models.utils import load_rwkv_with_peft

from hyperpyyaml import load_hyperpyyaml

def test_rwkv():

    # wget https://huggingface.co/BlinkDL/rwkv7-g1/resolve/main/rwkv7-g1d-0.1b-20260129-ctx8192.pth

    config_file = "./tests/yaml/test_rwkv.yaml"
    with open(config_file) as fin:
        cfg = load_hyperpyyaml(fin)

    llm = load_rwkv_with_peft(
        args = cfg["llm_args"],
        base_ckpt = "./tests/rwkv7-g1d-0.1b-20260129-ctx8192.pth",
    )