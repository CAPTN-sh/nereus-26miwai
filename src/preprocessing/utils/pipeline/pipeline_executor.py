from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from preprocessing.utils.pipeline.pipeline import Pipeline


class PipelineExecutor:

    def __init__(self, pipeline: Pipeline, tqdm_desc="Processing"):
        self.pipeline = pipeline
        self.task_iter, self.total = pipeline.load_tasks()
        self.desc = tqdm_desc

        self.results = []
        self.pending = deque()

    def run_parallel(self, max_workers=None):
        with self.pipeline.init_pool(max_workers) as executor:
            max_inflight = (max_workers or 8) * 8
            self.submit_task(executor, max_inflight)

            with tqdm(total=self.total, desc=self.desc) as bar:
                while self.pending:
                    future, key = self.pending.popleft()
                    try:
                        result = future.result()
                        if result is not None:
                            self.results.append(result)
                    except Exception as err:
                        print(f"Error processing {key}: {err}")
                    bar.update(1)
                    self.submit_task(executor, max(0, max_inflight - len(self.pending)))
                bar.refresh()

        return self.pipeline.save_results(self.results)

    def submit_task(self, executor: ProcessPoolExecutor, n_tasks=1):
        try:
            for _ in range(n_tasks):
                key, args = next(self.task_iter)
                future = executor.submit(self.pipeline.execut_task, key, *args)
                self.pending.append((future, key))
        except StopIteration:
            pass
