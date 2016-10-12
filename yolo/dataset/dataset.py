"""DataSet  base class 
"""
class DataSet(object):
  """Base DataSet
  """
  def __init__(self, common_params, dataset_params):
    """
    common_params: A params dict 
    dataset_params: A params dict
    """
    raise NotImplementedError

  def batch(self):
    """Get batch
    """
    raise NotImplementedError