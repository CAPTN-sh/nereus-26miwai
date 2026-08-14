from dataclasses import dataclass

from utils.config import STEPS_PER_MINUTE


@dataclass
class TraisformerARParams:
    """Hyperparameters for the autoregressive TrAISformer (Nguyen & Fablet, 2021).

    This is the original formulation: a causal transformer over discretised
    ``(x, y, SOG, COG)`` tokens, trained with next-token cross-entropy and decoded by
    rolling out one step at a time. Contrast with :class:`TraisformerParams`, which
    configures this repo's single-shot heatmap variant used as the NEREUS prior.
    """

    # Sequence lengths (obs_len / pred_len / max_dist are also read by AISDataModule)
    obs_len: int = 10 * STEPS_PER_MINUTE
    pred_len: int = 5 * STEPS_PER_MINUTE
    max_dist: int = 0        # no neighbour edges: this is a single-vessel model

    # Discretisation. pos_res is the side of a position cell in metres and puts a hard
    # floor of ~0.38 * pos_res on achievable ADE (25 m -> ~10 m, 50 m -> ~19 m), which a
    # perfectly-trained model still pays. Because x and y have separate softmaxes, the
    # parameter count grows only linearly as cells shrink -- but so does the number of
    # classes to learn, so finer is not automatically better.
    pos_res: int = 25

    # Per-attribute embedding widths; n_embd must equal their sum.
    n_x_embd: int = 96
    n_y_embd: int = 96
    n_sog_embd: int = 32
    n_cog_embd: int = 32
    n_embd: int = 256

    n_head: int = 8
    n_layer: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1

    # Decoding
    num_samples: int = 3          # K rollouts for the k_ade / k_fde columns
    # Row 0 of the rollout (behind the ade/fde columns) is greedy argmax by default,
    # matching the paper's "TrAISformer_No-Stoch" ablation. Set False to make it a
    # single stochastic sample instead (still one draw, not best-of-N -- k_ade/k_fde
    # already cover best-of-N). Eval-time only: safe to flip on a trained checkpoint,
    # no retraining needed.
    greedy_best: bool = True
    temperature: float = 1.0
    top_k: int | None = None      # None = sample from the full distribution
    sample_mode: str = "vicinity"  # "vicinity" (paper's pos_vicinity) | "full"
    r_vicinity: int = 8           # max cell displacement per step and axis
    #   Keep r_vicinity * pos_res at ~200 m: that is comfortably above the largest step
    #   in the Kiel data (139 m per 10 s) while stopping an under-trained model from
    #   teleporting across the map. Retune it whenever pos_res changes.
