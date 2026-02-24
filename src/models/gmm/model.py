import torch
from torch import nn
from models.traisformer.model import TrAISformer
from sklearn.mixture import GaussianMixture

class AIS_GMM(nn.Module):
    def __init__(self, prior_model: TrAISformer, n_clusters):
        super().__init__()
        self.prior_model = prior_model
        self.prior_model.eval()

        self.k = n_clusters
        self.gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            max_iter=200,
            n_init=5,
            reg_covar=1e-6,
        )

    def fit(self, train_loader, scene, device, max_samples):
        all_z = []
        total_samples = 0

        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)

                _, z = self.prior_model(batch, scene = scene)
                z = torch.nn.functional.normalize(z, dim=1)
                all_z.append(z.cpu())
                total_samples += z.size(0)

                if total_samples >= max_samples:
                    break

        z_cpu = torch.cat(all_z, dim=0).numpy()
        self.gmm.fit(z_cpu)

    def predict_proba(self, data, scene):
        """Soft cluster assignment p(k | z)."""
        with torch.no_grad():
            _, z = self.prior_model(data, scene)
            z = torch.nn.functional.normalize(z, dim=1)
            z = z.cpu().numpy()

        probs = self.gmm.predict_proba(z)
        return probs
    
class MAP_GMM(nn.Module):
    def __init__(self, gmm: AIS_GMM, cluster_maps):
        super().__init__()
        self.gmm = gmm
        self.cluster_maps = cluster_maps

    def forward(self, data, scene):
        cluster_prob = self.gmm.predict_proba(data, scene)
        cluster_prob = torch.tensor(cluster_prob, device=self.cluster_maps.device)

        density_map = torch.einsum(
            "bk,khw->bhw",
            cluster_prob,
            self.cluster_maps
        ).unsqueeze(1)
        
        return density_map, None