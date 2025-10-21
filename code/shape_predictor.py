import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from shape_utils import generate_edge_map, shape_loss
from dataset import create_BOLD5000_dataset, create_Kamitani_dataset
import torchvision.transforms as transforms
from PIL import Image


class ShapePredictor(nn.Module):
    """Lightweight network to predict shape/edge maps from fMRI signals"""

    def __init__(self, fmri_dim, shape_dim=320, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = fmri_dim // 2

        self.fmri_dim = fmri_dim
        self.shape_dim = shape_dim

        self.net = nn.Sequential(
            nn.Linear(fmri_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, shape_dim),
            nn.Sigmoid()  # Output probabilities for edge pixels
        )

    def forward(self, fmri):
        """
        Args:
            fmri: (batch, fmri_dim) - preprocessed fMRI signals
        Returns:
            shape_pred: (batch, shape_dim) - predicted edge/shape maps
        """
        return self.net(fmri)


class ShapeDataset(Dataset):
    """Dataset for training shape predictor: fMRI -> edge maps"""

    def __init__(self, dataset_name='BOLD5000', subjects=None, root_path='.',
                 edge_method='canny', max_samples=None):
        self.dataset_name = dataset_name
        self.edge_method = edge_method
        self.root_path = root_path

        # Create base dataset
        if dataset_name == 'BOLD5000':
            data_path = os.path.join(root_path, 'data/BOLD5000')
            subjects = subjects or ['CSI1']
            _, self.base_dataset = create_BOLD5000_dataset(
                data_path, patch_size=16, fmri_transform=torch.FloatTensor,
                image_transform=self._image_transform, subjects=subjects
            )
        elif dataset_name == 'GOD':
            data_path = os.path.join(root_path, 'data/Kamitani/npz')
            subjects = subjects or ['sbj_3']
            _, self.base_dataset = create_Kamitani_dataset(
                data_path, roi='VC', patch_size=16, fmri_transform=torch.FloatTensor,
                image_transform=self._image_transform, subjects=subjects
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Limit samples if specified
        if max_samples:
            self.length = min(len(self.base_dataset), max_samples)
        else:
            self.length = len(self.base_dataset)

        print(f"ShapeDataset created with {self.length} samples")

    def _image_transform(self, img):
        """Transform for images - keep as PIL for edge detection"""
        if not isinstance(img, Image.Image):
            # Handle different numpy array formats
            if isinstance(img, torch.Tensor):
                img = img.numpy()

            # Ensure proper shape and data type
            if img.ndim == 4:  # (1, H, W, C)
                img = img.squeeze(0)
            if img.ndim == 3 and img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                img = img.transpose(1, 2, 0)

            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                if img.max() <= 1.0:  # Normalized to [0,1]
                    img = (img * 255).astype(np.uint8)
                else:  # Already in [0,255] range
                    img = img.astype(np.uint8)

            img = Image.fromarray(img)

        # Resize to standard size for edge detection
        img = img.resize((256, 256))
        return img

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get fMRI and image from base dataset
        sample = self.base_dataset[idx]

        # Handle different return formats
        if isinstance(sample, dict):
            fmri = sample['fmri']
            image = sample['image']
        elif len(sample) == 2:
            fmri, image = sample
        else:
            fmri, image, _ = sample

        # Generate edge map from image
        image_np = np.array(image)
        edge_map = generate_edge_map(image_np, method=self.edge_method, flatten=True)
        edge_map = torch.tensor(edge_map, dtype=torch.float32)

        # Ensure edge map is 320-dim to match shape predictor output
        if edge_map.shape[0] == 256:
            # Pad to 320 dimensions
            edge_map = torch.cat([edge_map, torch.zeros(64)], dim=0)

        # Flatten fMRI if needed
        if fmri.dim() > 1:
            fmri = fmri.flatten()

        return fmri, edge_map


def train_shape_predictor(dataset_name='BOLD5000', subjects=None, num_epochs=50,
                         batch_size=16, lr=1e-3, save_path='shape_predictor.pth',
                         max_samples=None):
    """Train the shape predictor model"""

    print(f"Training shape predictor on {dataset_name} dataset...")

    # Create dataset
    dataset = ShapeDataset(dataset_name=dataset_name, subjects=subjects,
                          max_samples=max_samples)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Get fMRI dimension from first sample
    fmri_sample, _ = dataset[0]
    fmri_dim = fmri_sample.shape[0]

    print(f"fMRI dimension: {fmri_dim}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShapePredictor(fmri_dim=fmri_dim, shape_dim=320).to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (fmri, edge_maps) in enumerate(train_loader):
            fmri, edge_maps = fmri.to(device), edge_maps.to(device)

            optimizer.zero_grad()
            pred_edges = model(fmri)
            loss = F.mse_loss(pred_edges, edge_maps)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for fmri, edge_maps in val_loader:
                fmri, edge_maps = fmri.to(device), edge_maps.to(device)
                pred_edges = model(fmri)
                loss = F.mse_loss(pred_edges, edge_maps)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'fmri_dim': fmri_dim,
                'shape_dim': 320,
                'val_loss': val_loss,
                'dataset': dataset_name
            }, save_path)
            print(f"New best model saved with val loss: {val_loss:.4f}")

        scheduler.step()

    print(f"Training completed! Best model saved to {save_path}")
    return model


if __name__ == "__main__":
    # Quick test
    print("Testing shape predictor...")

    # Test model creation
    model = ShapePredictor(fmri_dim=1000, shape_dim=320)
    print(f"Model created: {model}")

    # Test forward pass
    dummy_fmri = torch.randn(4, 1000)
    output = model(dummy_fmri)
    print(f"Output shape: {output.shape}, range: [{output.min():.3f}, {output.max():.3f}]")

    print("✅ Shape predictor test passed!")