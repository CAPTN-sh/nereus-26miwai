from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Any
import pandas as pd
from concurrent.futures import Executor, ProcessPoolExecutor


class Pipeline(ABC):
    @abstractmethod
    def load_tasks(self) -> Tuple[Iterator[Tuple[Any, Tuple]], int]:
        """
        Load all df from disc and turn them into tasks.
        Return:
            - Iterator yielding (key, args)
            - Total number of tasks
        """
        pass

    @abstractmethod
    def execut_task(self, key, *args) -> pd.DataFrame:
        """
        Run a task. Is executed in parallel inside the "PipelineExecutor"
        """
        pass

    @abstractmethod
    def save_results(self, results: list[pd.DataFrame]) -> None:
        """
        Final transformation on the result that can not be done in parallel.
        Save the final result to disc.
        """
        pass

    def init_pool(self, max_workers: int) -> Executor:
        """
        Return a configured ProcessPoolExecutor.
        """
        return ProcessPoolExecutor(max_workers=max_workers)
