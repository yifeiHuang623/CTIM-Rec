# traj_lib/utils/dataloader/__init__.py
import importlib, pkgutil
from utils.logger import get_logger
from utils.register import VIEW_REGISTRY

logger = get_logger(__name__)

for mod in pkgutil.iter_modules(__path__, prefix=f"{__name__}."):
    before = set(VIEW_REGISTRY)
    importlib.import_module(mod.name)
    new_keys = set(VIEW_REGISTRY) - before
    for key in new_keys:
        logger.debug(f"View {key:>25s} found from {mod.name}")
