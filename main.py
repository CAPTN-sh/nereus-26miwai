from preprocessing.decoding.pipeline import DecodingPipeline
from preprocessing.feature_extraction.pileline import FeatureExtractionPipeline

if __name__ == "__main__":
    DecodingPipeline("C:/Users/Ben/shipwise/preprocessing/configs/config_decoder.yaml")
    FeatureExtractionPipeline(
        "C:/Users/Ben/shipwise/preprocessing/configs/config_feature_extractor.yaml"
    ).run()
