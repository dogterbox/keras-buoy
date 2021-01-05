from tensorflow.keras.callbacks import ModelCheckpoint
from .callbacks import EpochCounter, HistoryLogger
from .utils import merge_dicts_with_only_lists_as_values
import pickle
import os
import logging


class ModelTrainer(object):
  """Save and overwrite a model every 'period' epochs to 'to_path',
  preserving the number of epochs and the history dict over multiple interrupted
  executions.

  If to_path is mymodel.h5, then there will be mymodel_epoch_num.pkl and
  mymodel_history.pkl in the same directory as mymodel.h5, which hold backups for
  the epoch counter and the history dict, respectively.

  Args:
    period (int): How often to save the model and the accompanying
      parameters.
    to_path (str): A path to a model destination with the .h5 extension, which is
      where model weights will be saved.

  Returns: A Keras History.history dictionary of the entire training process.
  """
  def __init__(self, model, to_path, period=1,
               save_best_only=True, monitor='val_loss'):

    assert period > 0

    self.model = model
    self.period = period
    self.custom_objects = custom_objects
    self.to_path = to_path
    self.prefix = os.path.splitext(to_path)[0]
    self.checkpoint_file = self.prefix + "model.hdf5"
    self.epoch_num_file = self.prefix + "_epoch_num.pkl"
    self.history_file = self.prefix + "_history.pkl"
    self.save_best_only = save_best_only
    self.monitor = monitor

    # recover history
    self.history = self.get_history()
    # recover latest epoch
    self.initial_epoch = self.get_epoch_num()

  def _load_pickle(self, filePath, default_value=None):
    return pickle.load(open(filePath, 'rb')) if os.path.exists(filePath) else default_value

  def get_epoch_num(self):
    return self._load_pickle(self.epoch_num_file, 0)

  def get_history(self):
    return self._load_pickle(self.history_file, {})

  def _monitor_score(self, x_val, y_val, batch_size):
    if len(self.model.metrics_names) == 0:
      return None
    scores = self.model.evaluate(x_val, y_val, batch_size=batch_size)
    monitor = self.monitor.replace('val_', '')
    monitor_index = self.model.metrics_names.index(monitor)
    return scores[monitor_index]


  def _make_fit_args(self, *args, **kwargs):
    assert not 'initial_epoch' in kwargs
    logger = logging.getLogger()

    # add callbacks for periodic checkpointing
    if 'callbacks' not in kwargs:
      kwargs['callbacks'] = []

    hist_logger = HistoryLogger(period=self.period, history_path=self.history_file,
                                recovered_history=self.history)
    epoch_counter = EpochCounter(counter_path=self.epoch_num_file)
    model_checkpoint = ModelCheckpoint(self.to_path, save_best_only=self.save_best_only,
                                       verbose=True, period=self.period,
                                       monitor=self.monitor)
    if 'validation_data' in kwargs:
      # get batch size
      if 'validation_batch_size' in kwargs:
        batch_size = kwargs['validation_batch_size']
      elif:
        batch_size = kwargs['batch_size']
      else:
        batch_size = None

      # model evaluate
      score = self._monitor_score(*kwargs['validation_data'], batch_size=batch_size)
      if score is not None:
        model_checkpoint.best = score
      logger.info(f'checkpoint score: {model_checkpoint.best}')
    kwargs['callbacks'] += [hist_logger, epoch_counter, model_checkpoint]

    # Warn user if the training is already complete.
    if 'epochs' in kwargs and self.initial_epoch >= kwargs['epochs']:
      epochs = kwargs['epochs']
      logger.warning(f'You want to train for {epochs} epochs but {self.initial_epoch} epochs already completed; nothing to do.')
    return args, kwargs
  
  def _perform_final_save(self, remaining_history, epoch):
    # Combine histories and save
    combined_history = merge_dicts_with_only_lists_as_values([self.history, remaining_history.history])
    pickle.dump(combined_history, open(self.history_file, "wb"))
    # Dump last last epoch
    pickle.dump(epoch, open(self.epoch_num_file, "wb"))
    # Save model
    self.model.save(self.to_path)
    return combined_history
  
  def fit(self, *args, **kwargs):
    args, kwargs = self._make_fit_args(*args, **kwargs)
    remaining_history = self.model.fit(initial_epoch=self.initial_epoch, *args, **kwargs)
    combined_history = self._perform_final_save(remaining_history, epoch=kwargs['epochs'])
    return combined_history
  
  def fit_generator(self, *args, **kwargs):
    args, kwargs = self._make_fit_args(*args, **kwargs)
    remaining_history = self.model.fit_generator(initial_epoch=self.initial_epoch, *args, **kwargs)
    combined_history = self._perform_final_save(remaining_history, epoch=kwargs['epochs'])
    return combined_history
