from pathlib import Path

import optuna

from models.desire.model import DESIRE
from models.desire.nn.loss import loss_desire
from models.desire.utils.params import DESIREParams
from models.lstm.model import LSTMModel
from models.lstm.params import LSTMParams
from models.traisformer.loss import loss_intent_heatmap
from models.traisformer.model import TrAISformer
from models.traisformer.params import TraisformerParams
from train.eval import eval, eval_loss
from train.eval_heatmap import eval_heatmap
from train.train_local import tune_cpu
from utils.logger import logger

if __name__ == "__main__":
    data_folder = Path("data/ais/4_features/fhkiel_train/kiel/")
    scene_path = Path("data/scenes/fhkiel_train/kiel/bev.npz")
    scene_meta_path = Path("data/scenes/fhkiel_train/kiel/bev_meta.json")

    model_options = ["DESIRE", "LSTM", "TRAISFORMER"]
    model_choise = model_options[1]

    logger(file_prefix=f"train_local_{model_choise}")

    if model_choise == "DESIRE":
        model = DESIRE(DESIREParams())
        loss_fn = loss_desire
        eval_fn = eval

    if model_choise == "LSTM":
        model = LSTMModel
        model_params = LSTMParams()
        model_hyper_params = {"hidden_size": [32, 64, 128]}
        loss_fn = eval_loss
        eval_fn = eval

    if model_choise == "TRAISFORMER":
        cfg = TraisformerParams()
        model = TrAISformer(cfg)
        loss_fn = loss_intent_heatmap
        eval_fn = eval_heatmap

    # --- Optuna Study ---
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: tune_cpu(
            trial,
            model=model,
            model_params=model_params,
            model_hyper_params=model_hyper_params,
            loss_fn=loss_fn,
            eval_fn=eval_fn,
            data_folder=data_folder,
            scene_path=scene_path,
            scene_meta_path=scene_meta_path,
            num_epochs=10,
        ),
        n_trials=20,
    )

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
