from preprocessing.pipeline.pipeline_executor import PipelineExecutor
from preprocessing.steps.decoding.pipeline import DecodingPipeline
from preprocessing.steps.transform.edges.pipeline import EdgesPipeline
from preprocessing.steps.transform.nodes.pipeline import NodesPipeline
from utils.config import Config

if __name__ == "__main__":
    Config("configs/main.yaml")
    PipelineExecutor(DecodingPipeline()).run_parallel(max_workers=10)
    PipelineExecutor(NodesPipeline()).run_parallel(max_workers=20)
    PipelineExecutor(EdgesPipeline()).run_parallel(max_workers=20)
