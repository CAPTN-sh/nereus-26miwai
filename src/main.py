from preprocessing.steps.decoding.pipeline import DecodingPipeline
from preprocessing.steps.transform.nodes.pipeline import NodesPipeline
from preprocessing.steps.transform.edges.pipeline import EdgesPipeline
from preprocessing.utils.pipeline.pipeline_executor import PipelineExecutor
from utils.config import Config

if __name__ == "__main__":
    Config("src/preprocessing/configs/_main.yaml")
    PipelineExecutor(DecodingPipeline()).run_parallel()
    PipelineExecutor(NodesPipeline()).run_parallel()
    PipelineExecutor(EdgesPipeline()).run_parallel()
