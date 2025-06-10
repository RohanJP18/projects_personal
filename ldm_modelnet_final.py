import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import open3d as o3d

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def read_off(file_path):
    with open(file_path, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise ValueError("Invalid OFF file")
        n_verts, _, _ = map(int, f.readline().strip().split())
        verts = [list(map(float, f.readline().strip().split())) for _ in range(n_verts)]
    return np.array(verts)

def load_modelnet_pointclouds(root, samples=1024):
    pcs, prompts = [], []
    for category in os.listdir(root):
        for split in ['train', 'test']:
            cat_dir = os.path.join(root, category, split)
            if not os.path.isdir(cat_dir): continue
            for fname in os.listdir(cat_dir):
                if not fname.endswith('.off'): continue
                fpath = os.path.join(cat_dir, fname)
                try:
                    verts = read_off(fpath)
                    if verts.shape[0] < samples: continue
                    idx = np.random.choice(len(verts), samples, replace=False)
                    pc = verts[idx]
                    pcs.append(pc)
                    prompts.append(f"a 3d model of a {category}")
                except:
                    continue
    return pcs, prompts

class PointCloudDataset(Dataset):
    def __init__(self, pcs, prompts, tokenizer):
        self.pcs = pcs
        self.prompts = prompts
        self.tokenizer = tokenizer

    def __len__(self): return len(self.pcs)

    def __getitem__(self, idx):
        pc = torch.tensor(self.pcs[idx], dtype=torch.float32)
        tok = self.tokenizer(self.prompts[idx], return_tensors='pt', padding="max_length", truncation=True, max_length=77)
        return pc, tok['input_ids'].squeeze(0), tok['attention_mask'].squeeze(0)

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3*1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 3*1024))

    def forward(self, x):
        B = x.size(0)
        h = self.encoder(x.view(B, -1))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z).view(B, 1024, 3)
        return recon, mu, logvar

    def encode(self, x):
        return self.fc_mu(self.encoder(x.view(x.size(0), -1)))

class Diffusion(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, z_noisy, cond):
        return self.model(torch.cat([z_noisy, cond], dim=-1))

def chamfer(pc1, pc2):
    pc1, pc2 = pc1[:, None, :, :], pc2[:, :, None, :]
    dist = torch.norm(pc1 - pc2, dim=-1)
    return torch.mean(torch.min(dist, dim=2)[0]) + torch.mean(torch.min(dist, dim=1)[0])

def generate_from_prompt(prompt, vae, diffusion, tokenizer, clip_model):
    vae.eval(); diffusion.eval()
    with torch.no_grad():
        tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        input_ids = tokens["input_ids"].to(device)
        attn_mask = tokens["attention_mask"].to(device)
        cond = clip_model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state[:, 0, :]
        z = torch.randn(1, 128).to(device)
        z_gen = z - diffusion(z, cond)
        z_gen = torch.clamp(z_gen, -10, 10)
        z_gen = torch.nan_to_num(z_gen, nan=0.0)
        pc = vae.decoder(z_gen).view(1, 1024, 3)
        return pc[0].cpu().numpy()

def visualize_pointcloud(pc):
    pc = pc[~np.isnan(pc).any(axis=1)]
    pc -= np.mean(pc, axis=0)
    pc /= np.max(np.linalg.norm(pc, axis=1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])

def main():
    pcs, prompts = load_modelnet_pointclouds("ModelNet40")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    pcs_train, pcs_test, prompts_train, prompts_test = train_test_split(pcs, prompts, test_size=0.2)
    train_set = PointCloudDataset(pcs_train, prompts_train, tokenizer)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

    vae = VAE().to(device)
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)

    for epoch in range(5):
        for pc, _, _ in tqdm(train_loader, desc=f"VAE Epoch {epoch}"):
            pc = pc.to(device)
            recon, mu, logvar = vae(pc)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / pc.size(0)
            loss = F.mse_loss(recon, pc) + 0.001 * kl
            optimizer_vae.zero_grad(); loss.backward(); optimizer_vae.step()

    diffusion = Diffusion().to(device)
    optimizer_diff = torch.optim.Adam(diffusion.parameters(), lr=5e-5)

    for epoch in range(10):
        for pc, ids, mask in tqdm(train_loader, desc=f"Diffusion Epoch {epoch}"):
            pc, ids, mask = pc.to(device), ids.to(device), mask.to(device)
            with torch.no_grad():
                z = vae.encode(pc)
                cond = clip_model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
            noise = torch.randn_like(z)
            z_noisy = z + noise
            z_noisy = torch.clamp(z_noisy, -10, 10)
            pred = diffusion(z_noisy, cond)
            loss = F.mse_loss(pred, noise)
            optimizer_diff.zero_grad(); loss.backward(); optimizer_diff.step()

    print("\n✔️ Training complete. Now you can enter a custom prompt.")
    prompt = input("Enter your 3D shape prompt (e.g., 'a 3d model of a bed'): ")
    recon_pc = generate_from_prompt(prompt, vae, diffusion, tokenizer, clip_model)
    print("\n🔍 Visualizing generated point cloud...")
    visualize_pointcloud(recon_pc)
    np.savetxt("generated_pointcloud.xyz", recon_pc)
    print("Saved to generated_pointcloud.xyz")

if __name__ == "__main__":
    main()
