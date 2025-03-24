from .BuildModel import TeacherModel,StudentModel
from .DataAugmentation import mixup_data, cutmix_data
from .Dataset import Dataset
from .Loss import LabelSmoothingCrossEntropy, CRDLoss, DistillationLoss
from .TrainModel import TrainModel
from . import utils
# from importlib import import_module
# import os


# package_dir = os.path.dirname(__file__)

# for filename in os.listdir(package_dir):
#     if filename.endswith(".py") and filename != "__init__.py":
#         moudle_name = filename[:-3]
#         import_module(f".{moudle_name}",package = __name__)

__all__ = ["TeacherModel", "StudentModel", "mixup_data", "cutmix_data",
           "Dataset", "LabelSmoothingCrossEntropy", "CRDLoss", "DistillationLoss",
           "TrainModel", "utils"]