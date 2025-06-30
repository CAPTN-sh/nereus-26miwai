from preprocessing.decoding.pipeline import DecodingPipeline
from preprocessing.feature_extraction.pileline import FeatureExtractionPipeline
from preprocessing.utils.config import load_config

if __name__ == "__main__":
    paths = load_config("preprocessing/configs/main_config.yaml")["paths"]
    DecodingPipeline(paths["decoder_config"]).run()
    FeatureExtractionPipeline(paths["feature_extractor_config"]).run()
