"""
Test code for active pipeline:

"""
import os
from absl.testing import absltest
from absl.testing import parameterized
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime
import wandb
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(os.path.abspath(__file__)).resolve().parents[2]
sys.path.append(os.path.join(str(BASE_DIR), 'scripts'))
sys.path.append(os.path.join(str(BASE_DIR), 'active_pipeline'))
import train_hydra
from  call_active_loop import call_active_all


ACTIVE_LOOP = BASE_DIR / "active_pipeline/configs/config_ibl_active.yaml"
IBL_EXPERIMENT = BASE_DIR / "active_pipeline/configs/config_ibl_experiment.yaml"

eval_configs = {
  'active_loop': ACTIVE_LOOP,
  'toy_experiment': IBL_EXPERIMENT,
}


def check_model_eq(model, model2):
  for param1, param2 in zip(model.parameters(), model2.parameters()):
    if param1.data.ne(param2.data).sum() > 0:
      return False
  return True

def create_cfg(eval_config_name) -> dict:
  """Load all toy data config file without hydra."""
  cfg = yaml.load(open(str(eval_configs[eval_config_name])), Loader=yaml.FullLoader)
  return OmegaConf.create(cfg)


class TestNaiveRun(parameterized.TestCase):
  @parameterized.named_parameters(
    ("base_toy_run", "toy_experiment"),
  )
  def test_imbalanced_losses(self, config_name):
    # Read cfg
    cfg = create_cfg(config_name)
    cfg.training.fast_dev_run = True
    cfg.wandb.logger = False
    cwd = os.getcwd()
    today_str = datetime.now().strftime("%y-%m-%d")
    ctime_str = datetime.now().strftime("%H-%M-%S")
    new_dir = f"./outputs/{today_str}/{ctime_str}"  # _active_iter_{str(current_iteration)}"
    os.makedirs(new_dir, exist_ok=False)
    os.chdir(new_dir)
    train_output_dir = train_hydra.train(cfg)
    os.chdir(cwd)
    if cfg.wandb.logger:
      wandb.finish()
    return train_output_dir

  @parameterized.named_parameters(
    ("base_active_run", "active_loop"),
  )
  def test_active_run(self, config_name):
    # read config
    cfg = create_cfg(config_name)
    # add debugging config
    OmegaConf.set_struct(cfg, True)
    OmegaConf.update(cfg, "active_pipeline.experiment_kwargs.training.fast_dev_run", True, force_add=True)
    OmegaConf.update(cfg, "active_pipeline.experiment_kwargs.wandb.logger", False, force_add=True)
    cfg.active_pipeline.start_iteration = 0
    cfg.active_pipeline.end_iteration = 1
    cfg.iteration_0.use_seeds = [0]
    cfg.iteration_1.use_seeds = [0]

    new_active_cfg = call_active_all(cfg)
    self.assertEqual(type(new_active_cfg), DictConfig)


if __name__ == "__main__":
  absltest.main()