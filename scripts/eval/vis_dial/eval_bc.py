import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import hydra
from omegaconf import DictConfig, OmegaConf
import visdial.load_objects
from inference import inference

@hydra.main(config_path="../../../config/socratic_dial", config_name="eval_bc")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    inference(cfg)

if __name__ == "__main__":
    main()
