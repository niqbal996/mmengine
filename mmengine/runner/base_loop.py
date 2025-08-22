# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Union

from torch.utils.data import DataLoader


class BaseLoop(metaclass=ABCMeta):
    """Base loop class.

    All subclasses inherited from ``BaseLoop`` should overwrite the
    :meth:`run` method.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
    """

    def __init__(self, runner, dataloader: Union[DataLoader, Dict]) -> None:
        self._runner = runner
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.dataloader = runner.build_dataloader(
                dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader = dataloader

    @property
    def runner(self):
        return self._runner

    @abstractmethod
    def run(self) -> Any:
        """Execute loop."""

class ActiveBaseLoop(BaseLoop):
    """Base loop class for active learning with dual dataloaders.

    Inherits from BaseLoop, but expects two dataloaders: source and target.
    """

    def __init__(self, runner, dataloader: dict) -> None:
        # Expect dataloader to be a dict with 'dataloader_source' and 'dataloader_target'
        dataloader_source = dataloader.get('dataloader_source')
        dataloader_target = dataloader.get('dataloader_target')
        # Optionally, pass one of them to BaseLoop for compatibility
        super().__init__(runner, dataloader_source)
        # Build both dataloaders
        if isinstance(dataloader_source, dict):
            diff_rank_seed = runner._randomness_cfg.get('diff_rank_seed', False)
            self.dataloader_source = runner.build_dataloader(
                dataloader_source, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader_source = dataloader_source

        if isinstance(dataloader_target, dict):
            diff_rank_seed = runner._randomness_cfg.get('diff_rank_seed', False)
            self.dataloader_target = runner.build_dataloader(
                dataloader_target, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader_target = dataloader_target

    @property
    def runner(self):
        return self._runner

    @abstractmethod
    def run(self) -> Any:
        """Execute loop."""
