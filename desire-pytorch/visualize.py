import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch

from desire.data.loader import data_loader
from desire.utils.misc import relative_to_abs
from desire.models import DESIRE
from desire.utils.params import IOCParams, SGMParams

def main():
    # Config
    restore_model_path = 'weights/iter_005.pth'
    path_of_static_image = 'bg.png'
    train_path = './dataset/denmark/test'
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load scene image
    img = mpimg.imread(path_of_static_image)
    image = Image.open(path_of_static_image)
    width, height = image.size
    scene = TF.to_tensor(image).unsqueeze(0).to(device)

    # Load model
    desire = DESIRE(IOCParams(), SGMParams()).to(device)
    state_dict_checkpoint = torch.load(restore_model_path, map_location=device)
    desire.load_state_dict(state_dict_checkpoint)
    desire.eval()

    # Load one random batch (scene)
    train_dset, train_loader = data_loader(train_path, batch_size=batch_size, delim=",")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    for n in range(1):

        sample = next(iter(train_loader))
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _, _, seq_start_end) = sample

        obs_traj = obs_traj.permute(1,2,0)
        pred_traj_gt = pred_traj_gt.permute(1,2,0)

        obs_traj_rel = obs_traj_rel.permute(1,2,0)
        pred_traj_gt_rel = pred_traj_gt_rel.permute(1,2,0)

        x_start = obs_traj[:, :, 0].to(device)

        # Inference
        with torch.no_grad():
            y_pred_traj, pred_deta = desire.inference(obs_traj_rel, scene, x_start, seq_start_end)
        # Convert to absolute positions
        pred_abs = relative_to_abs(y_pred_traj*10, obs_traj[:, :, -1])  # (1, 2, 12)

        def map_img(traj):
            return traj[0]*width/10, height-traj[1]*height/10
        
        for i in range(obs_traj.size(0)):
            ax.scatter(*obs_traj[i], color='blue', alpha=0.6)
            ax.scatter(*pred_abs[i], color='red', alpha=0.6)
            ax.scatter(*pred_traj_gt[i], color='green', alpha=0.6)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Observed'),
        Line2D([0], [0], color='red', lw=2, label='Predicted'),
        Line2D([0], [0], color='green', lw=2, label='Ground Truth')
    ]
    ax.legend(handles=legend_elements)


    ax.set_title("Trajectory Visualization on Scene")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # optional but good on Windows
    main()