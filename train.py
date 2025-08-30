import os
import glob
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from model import CNNTransformer3D  # hybrid CNN+Transformer model

# fallback: use point-wise MSE loss instead of ChamferDistance
criterion = torch.nn.MSELoss()

# helper to read .dat vertices (skip non-numeric header)
def read_dat(path):
    verts = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                verts.append([x, y, z])
            except ValueError:
                continue
    return torch.tensor(verts, dtype=torch.float)


class MultiViewShapeNetDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # mesh file has same basename in rendering folder
        mesh_path = os.path.splitext(img_path)[0] + '.dat'
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"Expected mesh .dat not found: {mesh_path}")
        # parse .dat manually to avoid loadtxt errors
        verts = read_dat(mesh_path)
        return img, verts


def main():
    root = '/mnt/drive/aditya22040/god3/pixel2mesh/pixel2mesh/ShapeNetP2M/02691156'
    image_paths = []

    # Recursively collect all .png files with a corresponding .dat
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.png'):
                img_path = os.path.join(dirpath, filename)
                dat_path = img_path.replace('.png', '.dat')
                if os.path.exists(dat_path):
                    image_paths.append(img_path)

    print(f"Total valid image-mesh pairs: {len(image_paths)}")
    # print(f"Sample image path: {image_paths[0]}")
    train_imgs, val_imgs = train_test_split(image_paths, test_size=0.1, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = MultiViewShapeNetDataset(train_imgs, transform)
    val_ds = MultiViewShapeNetDataset(val_imgs, transform)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNTransformer3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for img, verts in train_loader:
            img = img.to(device)
            verts_gt = verts.to(device)
            preds = model(img)  # (B, M, 3)

            # subsample if needed
            if verts_gt.size(1) != preds.size(1):
                idx = torch.randperm(verts_gt.size(1))[:preds.size(1)]
                verts_sub = verts_gt[:, idx, :]
            else:
                verts_sub = verts_gt

            loss = criterion(preds, verts_sub)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, verts in val_loader:
                img = img.to(device)
                verts_gt = verts.to(device)
                preds = model(img)
                if verts_gt.size(1) != preds.size(1):
                    idx = torch.randperm(verts_gt.size(1))[:preds.size(1)]
                    verts_sub = verts_gt[:, idx, :]
                else:
                    verts_sub = verts_gt
                val_loss += criterion(preds, verts_sub).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{epochs} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), 'cnn_transformer3d_mse.pt')


if __name__ == '__main__':
    main()
