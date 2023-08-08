import os
import numpy as np
from typing import Any, Dict, Optional, Union
import lightning.pytorch as pl


def add_to_logger(logger, error_df, error_name, preds_file, index):
  if logger is None or (logger is not pl.loggers.WandbLogger):
    return
  preds_name = os.path.basename(preds_file).replace(".csv", "")
  with_set = True if 'set' in error_df.columns else False
  if with_set:
    keypoint_names = error_df.columns.drop('set').values
  else:
    keypoint_names = error_df.columns.values
  if error_name == 'temporal_norm':
    error_df['directory'] = preds_file.rsplit('/',1)[-1].replace(".csv","")
  else:
    error_df['directory'] = index.str.split('/').str[-2]

  if with_set:
    grouped = error_df.groupby(['directory', 'set'])
  else:
    grouped = error_df.groupby(['directory'])

  for group_id, group in grouped:
    if with_set:
      directory, set_name = group_id[0], group_id[1]
    else:
      directory, set_name = group_id[0], 'all'

    error_value = group[keypoint_names].values
    mean_error = np.nanmean(error_value)

    logger.experiment.log(
      {'extra_metrics/{}/{}'.format(error_name, directory): mean_error,
       #'extra_metrics/directory': directory,
       #'video': directory,
       #'pred_file': preds_name,
       #'set': set_name,
       #'bodypart': 'all'
       })

    """
    mean_indiv_error = np.nanmean(error_value, axis=0)
    for ii, keypoint_name in enumerate(keypoint_names):
      logger.experiment.log(
        {'extra_metrics/{}'.format(error_name): mean_indiv_error[ii],
         'video': directory, 'pred_file': preds_name,
         'bodypart': keypoint_name, 'set': set_name})
    """
  return
