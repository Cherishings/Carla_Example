import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
from torchsummary import summary
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import datetime

# 生成时间戳
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

save_dir = os.path.expanduser(f"~/Xi/code/Transformer/Transformer_Full_result_{timestamp}")
os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在就创建

best_model_path = os.path.join(save_dir, "transformer_best_model.pth")
final_model_path = os.path.join(save_dir, "transformer_corrected_VehicleNNController_NAG.pth")
training_csv_path = os.path.join(save_dir, "transformer_corrected_training_performance_NAG.csv")
loss_plot_path = os.path.join(save_dir, "transformer_loss_plot.png")
seg_acc_plot_path = os.path.join(save_dir, "transformer_seg_accuracy_plot.png")

# ----------------------------
# Device Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"Using device: {device}")

# ----------------------------
# Segmentation Settings
# ----------------------------
# Class mapping: Road, Cars, Lane Marks
# Define class mapping
CLASS_MAP = {
    1: 1,   # Road
    14: 2,  # Cars
    24: 3,   # Road Lane Marks
    21: 4, # Obstacles
    12: 5, # Pedestrians
    6: 6 # Traffic lights
}
BACKGROUND_LABEL = 0  # Everything else becomes background
NUM_CLASSES = 7       # (Background + Road + Cars + Lane Mark, etc)

def remap_mask(mask, class_map, background_label=0):
    """Remaps mask pixel values based on class_map."""
    remapped = np.full_like(mask, background_label)
    for old_class, new_class in class_map.items():
        remapped[mask == old_class] = new_class
    return remapped

# ----------------------------
# EMA Normolization
# ----------------------------
class LossEMA:
    def __init__(self, alpha=0.1, eps=1e-8):
        self.alpha = alpha
        self.eps = eps
        self.ema = {}

    def update(self, name, value):
        if name not in self.ema:
            self.ema[name] = value
        else:
            self.ema[name] = self.alpha * value + (1 - self.alpha) * self.ema[name]
        return self.ema[name]

    def normalize(self, name, value):
        ema_val = self.ema.get(name, value)
        return value / (ema_val + self.eps)

# ----------------------------
# output 30 waypoints
# ----------------------------
def equal_spacing_route(points, num_points=30):
    route = np.concatenate((np.zeros_like(points[:1]), points))  # prepend zero
    shift = np.roll(route, 1, axis=0)
    shift[0] = shift[1]
    dists = np.linalg.norm(route - shift, axis=1)
    dists = np.cumsum(dists)
    dists += np.arange(0, len(dists)) * 1e-4
    x = np.linspace(0, dists[-1], num=num_points)
    interp_points = np.stack([
        np.interp(x, dists, route[:, 0]),
        np.interp(x, dists, route[:, 1])
    ], axis=1)
    return interp_points


# ----------------------------
# Data Augmentation & Transforms
# ----------------------------
transform = A.Compose([
    # A.Resize(256, 256),
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
],
    additional_targets={"depth": "mask"}
)

def crop_array(images_i):  # images_i must have dimensions (H,W,C) or (H,W), please note other models have been trained without cropping !!!
    """
    Crop rgb images to the desired height and width
    """
    cropped_height = 384  # crops off the bottom part
    cropped_width = 1024  # crops off both sides symmetrically
    assert cropped_height <= images_i.shape[0]
    assert cropped_width <= images_i.shape[1]
    side_crop_amount = (images_i.shape[1] - cropped_width) // 2
    if len(images_i.shape) > 2:  # for rgb, we have 3 channels
        return images_i[0:cropped_height, side_crop_amount:images_i.shape[1] - side_crop_amount, :]
    else:  # for depth and semantics, there is no channel dimension
        return images_i[0:cropped_height, side_crop_amount:images_i.shape[1] - side_crop_amount]

# ----------------------------
# Custom Multi-Task Dataset
# ----------------------------
class MultiTaskDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, num_depth_bins=8):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.num_depth_bins = num_depth_bins 
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Image and semantic mask paths
        img_path = os.path.join(self.image_dir, row["Image_Path"])
        mask_path = os.path.join(self.image_dir, row["Semantic_Path"])
        depth_path = os.path.join(self.image_dir, row["Depth_Path"])
        
        # Load image and convert to RGB
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask as grayscale and remap classes
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask at {mask_path}")

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint8: [0,255]
        if depth is None:
            raise ValueError(f"Failed to read depth at {depth_path}")
        
        image = crop_array(image)

        mask = crop_array(mask)
        mask = remap_mask(mask, CLASS_MAP, BACKGROUND_LABEL)

        depth = crop_array(depth)
        depth = (depth.astype(np.float32) / 255.0 * (self.num_depth_bins - 1)).astype(np.uint8)
          
        if self.transform:
            augmented = self.transform(image=image, mask=mask, depth=depth)
            image = augmented["image"]          # [3, H, W]
            mask = augmented["mask"]            # [H, W] (single channel)
            depth = augmented["depth"]          # [H, W]
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            mask = torch.tensor(mask, dtype=torch.long)
            depth = torch.tensor(depth, dtype=torch.long)
            
        
        # Regression targets:
        # Target waypoint (2 values)
        target_waypoint_np = row[["Target_Point_X", "Target_Point_Y"]].values.astype(np.float32)
        target_waypoint = torch.from_numpy(target_waypoint_np)
        # Waypoints (10 values: WX1, WY1, WX2, WY2, WX3, WY3, WX4, WY4, WX5, WY5)
        wp_columns = ["WX1", "WY1", "WX2", "WY2", "WX3", "WY3", "WX4", "WY4", "WX5", "WY5"]
        waypoints_np = row[wp_columns].values.astype(np.float32).reshape(5, 2)  # numpy: shape [5, 2]

        # 插值为 30 个点
        waypoints_interp = equal_spacing_route(waypoints_np, num_points=30)
        waypoints = torch.from_numpy(waypoints_interp).float()
        
        # Brake and target speed (regression targets)
        brake = np.float32(row["Brake"])
        target_speed = np.float32(row["Target_Speed"])
        speed = torch.tensor(row["Speed"], dtype=torch.float32)
        waypoints = waypoints.clone().detach()
        brake = torch.tensor(row["Brake"], dtype=torch.float32)
        target_speed = torch.tensor(row["Target_Speed"], dtype=torch.float32)
        
        return image, mask, depth, target_waypoint, waypoints, brake, target_speed, speed


# ----------------------------
# Load and Shuffle Dataset
# ----------------------------
data_path = "/home/carla/Xi/code/Transformer/Corrected_WP_Data_depth.xlsx"  # Excel file path
df = pd.read_excel(data_path)
df = shuffle(df).reset_index(drop=True)

# Define image_dir (if paths in Excel are absolute, you can leave this as empty string)
image_dir = ""  

# Split data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = MultiTaskDataset(train_df, image_dir, transform=transform)
val_dataset = MultiTaskDataset(val_df, image_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)

# ----------------------------
# Define the VehicleNNController Model
# ----------------------------
class VehicleTransformerMAnet(nn.Module):
    def __init__(self, num_classes=7, num_waypoints=30, d_model=128, nhead=8, num_layers=3):
        super(VehicleTransformerMAnet, self).__init__()
        self.num_waypoints = num_waypoints

        # --- Encoder Backbone (mit_b2 from SegFormer) ---
        self.encoder = smp.Unet(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )

        # Visual feature projection
        self.visual_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),     # [B, 512, H, W] -> [B, 512, 1, 1]
            nn.Flatten(),                     # -> [B, 512]
            nn.Linear(512, d_model),          # -> [B, d_model]
            nn.ReLU()
        )

        self.input_proj = nn.Conv2d(512, d_model, kernel_size=1)

        # Meta data processing: target_point (2) + speed (1)
        self.meta_fc = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU()
        )

        # --- Transformer Decoder for Waypoints ---
        self.query_embed = nn.Parameter(torch.randn(num_waypoints, d_model))  # [num_waypoints, d_model]
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.waypoint_head = nn.Linear(d_model, 2)  # Output: x, y

        # --- Depth head ---
        self.depth_head = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=1)
        )

        # --- Control heads ---
        self.speed_head = nn.Linear(d_model, 1)
        self.brake_head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x_img, target_point, current_speed):
        # Encoder
        features = self.encoder.encoder(x_img)

        # print(f"Feature length: {len(features)}")   # << 添加这个
        decoder_out = self.encoder.decoder(features)
        seg_out = self.encoder.segmentation_head(decoder_out)

        # Depth prediction
        depth_out = self.depth_head(decoder_out)

        # Visual feature
        last_feat = features[-1]  # [B, 512, H, W]
        visual_feat = self.visual_fc(last_feat)  # [B, d_model]

        # Meta info
        meta_input = torch.cat([target_point, current_speed.unsqueeze(1)], dim=1)  # [B, 3]
        meta_feat = self.meta_fc(meta_input)  # [B, d_model]

        
        # Waypoint memory 使用 spatial + meta
        x = self.input_proj(features[-1])
        memory = x.flatten(2).permute(0, 2, 1)
        memory = torch.cat([meta_feat.unsqueeze(1), memory], dim=1)

        # Waypoint decoder
        queries = self.query_embed.unsqueeze(0).expand(x_img.size(0), -1, -1)
        decoded = self.transformer_decoder(tgt=queries, memory=memory)
        waypoints = self.waypoint_head(decoded)

        # 视觉 + meta
        control_feat = visual_feat + meta_feat
        speed = self.speed_head(control_feat)
        brake = self.brake_head(control_feat)

        return seg_out, waypoints, speed, brake, depth_out
# Instantiate and move the model to device
#model = VehicleTransformerMAnet(num_classes=NUM_CLASSES, debug = True).to(device)
model = VehicleTransformerMAnet(num_classes=NUM_CLASSES).to(device)

# ----------------------------
# Loss Functions & Optimizer
# ----------------------------
criterion_seg = nn.CrossEntropyLoss()
criterion_wp = nn.MSELoss()
criterion_speed = nn.MSELoss()
criterion_brake = nn.BCELoss()
criterion_depth = nn.CrossEntropyLoss() 

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# ----------------------------
# Helper Function to Compute Semantic Accuracy
# ----------------------------
def compute_seg_accuracy(pred, target):
    # pred: [B, num_classes, H, W]
    # target: [B, H, W] (values 0..num_classes-1)
    pred_labels = torch.argmax(pred, dim=1)
    correct = (pred_labels == target).float().sum()
    total = torch.numel(target)
    return (correct / total).item()

def compute_depth_accuracy(pred, target):
    # pred: [B, num_classes, H, W]
    # target: [B, H, W] (values 0..num_classes-1)
    pred_labels = torch.argmax(pred, dim=1)
    correct = (pred_labels == target).float().sum()
    total = torch.numel(target)
    return (correct / total).item()

# ----------------------------
# Training Loop
# ----------------------------
print("Starting Training")
num_epochs = 17
training_stats = []
# ALL_LOSS
train_losses = []
val_losses = []
# Semantic Segmentation Accuracy
train_seg_acc = []
val_seg_acc = []
# Depth Classification Accuracy
train_depth_acc = []
val_depth_acc = []

# ----------------------------
# individual losses
# ----------------------------
train_loss_seg = [] # semantic segmentation loss
train_loss_depth = [] # depth classification loss
train_loss_wp = [] # waypoint prediction loss
train_loss_speed = [] # speed prediction loss
train_loss_brake = [] # brake prediction loss

val_loss_seg = []   
val_loss_depth = []
val_loss_wp = []
val_loss_speed = []
val_loss_brake = []

best_val_loss = float("inf")
loss_ema = LossEMA(alpha=0.1)
best_epoch = -1

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_depth_acc = 0.0
    running_seg_acc = 0.0
    total_real_loss = 0.0
    # Individual losses
    total_seg_loss = 0.0
    total_depth_loss = 0.0
    total_wp_loss = 0.0
    total_speed_loss = 0.0
    total_brake_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        images, seg_masks, depth_labels, target_waypoints, waypoints, brakes, target_speeds, speeds = batch
        images = images.to(device)
        seg_masks = seg_masks.to(device).long()
        target_waypoints = target_waypoints.to(device)
        waypoints = waypoints.to(device)
        brakes = brakes.to(device).float()
        target_speeds = target_speeds.to(device).float()
        speeds = speeds.to(device).float()
        depth_labels = depth_labels.to(device).long()

        optimizer.zero_grad()
        teacher_forcing_ratio = max(0.1, 1.0 - epoch * 0.1)
        model._last_gt_waypoints = waypoints
        semantic_pred, wp_pred, speed_pred, brake_pred, depth_pred = model(images, target_waypoints, speeds)

        loss_seg = criterion_seg(semantic_pred, seg_masks)
        loss_wp = criterion_wp(wp_pred, waypoints)
        loss_speed = criterion_speed(speed_pred, target_speeds.view(-1, 1))
        loss_brake = criterion_brake(brake_pred, brakes.view(-1, 1))
        loss_depth = criterion_depth(depth_pred, depth_labels)

        #loss = 0.2 * loss_seg + 0.6 * loss_wp + 0.1 * loss_speed + 0.1 * loss_brake
        #loss = 0.15 * loss_seg + 0.15 * loss_depth + 0.5 * loss_wp + 0.1 * loss_speed + 0.1 * loss_brake

        # === update ===
        loss_ema.update("seg", loss_seg.item())
        loss_ema.update("depth", loss_depth.item())
        loss_ema.update("wp", loss_wp.item())
        loss_ema.update("speed", loss_speed.item())
        loss_ema.update("brake", loss_brake.item())

        #  === Normolization ===
        norm_seg = loss_ema.normalize("seg", loss_seg)
        norm_depth = loss_ema.normalize("depth", loss_depth)
        norm_wp = loss_ema.normalize("wp", loss_wp)
        norm_speed = loss_ema.normalize("speed", loss_speed)
        norm_brake = loss_ema.normalize("brake", loss_brake)

        loss = (0.7 * norm_wp + 0.1 * norm_seg + 0.1 * norm_depth + 0.05 * norm_speed + 0.05 * norm_brake)
        real_loss = 0.7 * loss_wp + 0.1 * loss_seg + 0.1 * loss_depth + 0.05 * loss_speed + 0.05 * loss_brake
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_real_loss += real_loss.item()
        running_seg_acc += compute_seg_accuracy(semantic_pred, seg_masks)
        running_depth_acc += compute_depth_accuracy(depth_pred, depth_labels)
        # Individual losses
        total_seg_loss += loss_seg.item()
        total_depth_loss += loss_depth.item()
        total_wp_loss += loss_wp.item()
        total_speed_loss += loss_speed.item()
        total_brake_loss += loss_brake.item()

    avg_train_loss = running_loss / len(train_loader)
    avg_real_train_loss = total_real_loss / len(train_loader)
    avg_train_acc = running_seg_acc / len(train_loader)
    avg_train_depth_acc = running_depth_acc / len(train_loader)
   
    # Append individual losses to lists
    train_loss_seg.append(total_seg_loss / len(train_loader))
    train_loss_depth.append(total_depth_loss / len(train_loader))
    train_loss_wp.append(total_wp_loss / len(train_loader))
    train_loss_speed.append(total_speed_loss / len(train_loader))
    train_loss_brake.append(total_brake_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0.0
    total_real_val_loss = 0.0
    val_seg_acc_total = 0.0
    val_depth_acc_total = 0.0
    # Reset individual losses for validation
    val_seg_loss = 0.0
    val_depth_loss = 0.0
    val_wp_loss = 0.0
    val_speed_loss = 0.0
    val_brake_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images, seg_masks, depth_labels, target_waypoints, waypoints, brakes, target_speeds, speeds = batch
            images = images.to(device)
            seg_masks = seg_masks.to(device).long()
            target_waypoints = target_waypoints.to(device)
            waypoints = waypoints.to(device)
            brakes = brakes.to(device).float()
            target_speeds = target_speeds.to(device).float()
            speeds = speeds.to(device).float()
            depth_labels = depth_labels.to(device).long()

            semantic_pred, wp_pred, speed_pred, brake_pred, depth_pred = model(images, target_waypoints, speeds)
            loss_seg = criterion_seg(semantic_pred, seg_masks)
            loss_wp = criterion_wp(wp_pred, waypoints)
            loss_speed = criterion_speed(speed_pred, target_speeds.view(-1, 1))
            loss_brake = criterion_brake(brake_pred, brakes.view(-1, 1))
            loss_depth = criterion_depth(depth_pred, depth_labels)
            #loss = 0.2 * loss_seg + 0.6 * loss_wp + 0.1 * loss_speed + 0.1 * loss_brake
            #loss = 0.15 * loss_seg + 0.15 * loss_depth + 0.5 * loss_wp + 0.1 * loss_speed + 0.1 * loss_brake

            #  === Normolization ===
            norm_seg = loss_ema.normalize("seg", loss_seg)
            norm_depth = loss_ema.normalize("depth", loss_depth)
            norm_wp = loss_ema.normalize("wp", loss_wp)
            norm_speed = loss_ema.normalize("speed", loss_speed)
            norm_brake = loss_ema.normalize("brake", loss_brake)

            loss = (0.7 * norm_wp + 0.1 * norm_seg + 0.1 * norm_depth + 0.05 * norm_speed + 0.05 * norm_brake)
            real_loss = 0.7 * loss_wp + 0.1 * loss_seg + 0.1 * loss_depth + 0.05 * loss_speed + 0.05 * loss_brake
            
            val_loss += loss.item()
            total_real_val_loss += real_loss.item()
            val_seg_acc_total += compute_seg_accuracy(semantic_pred, seg_masks)
            val_depth_acc_total += compute_depth_accuracy(depth_pred, depth_labels)
            # Individual losses
            val_seg_loss += loss_seg.item()
            val_depth_loss += loss_depth.item()
            val_wp_loss += loss_wp.item()
            val_speed_loss += loss_speed.item()
            val_brake_loss += loss_brake.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_real_val_loss = total_real_val_loss / len(val_loader)
    avg_val_acc = val_seg_acc_total / len(val_loader)
    avg_val_depth_acc = val_depth_acc_total / len(val_loader)
    # Append individual validation losses to lists
    val_loss_seg.append(val_seg_loss / len(val_loader))
    val_loss_depth.append(val_depth_loss / len(val_loader))
    val_loss_wp.append(val_wp_loss / len(val_loader))
    val_loss_speed.append(val_speed_loss / len(val_loader))
    val_loss_brake.append(val_brake_loss / len(val_loader))


    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_seg_acc.append(avg_train_acc)
    val_seg_acc.append(avg_val_acc)
    train_depth_acc.append(avg_train_depth_acc)
    val_depth_acc.append(avg_val_depth_acc)

    # Print epoch statistics
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Normalized Loss -> Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
    print(f"  Real Loss       -> Train: {avg_real_train_loss:.4f}, Val: {avg_real_val_loss:.4f}")
    print(f"  SegLoss         -> Train: {train_loss_seg[-1]:.4f}, Val: {val_loss_seg[-1]:.4f}")
    print(f"  DepthLoss       -> Train: {train_loss_depth[-1]:.4f}, Val: {val_loss_depth[-1]:.4f}")
    print(f"  WPLoss          -> Train: {train_loss_wp[-1]:.4f}, Val: {val_loss_wp[-1]:.4f}")
    print(f"  SpeedLoss       -> Train: {train_loss_speed[-1]:.4f}, Val: {val_loss_speed[-1]:.4f}")
    print(f"  BrakeLoss       -> Train: {train_loss_brake[-1]:.4f}, Val: {val_loss_brake[-1]:.4f}")
    print(f"  SegAcc          -> Train: {avg_train_acc:.4f}, Val: {avg_val_acc:.4f}")
    print(f"  DepthAcc        -> Train: {avg_train_depth_acc:.4f}, Val: {avg_val_depth_acc:.4f}")

    training_stats.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "real_train_loss": avg_real_train_loss,
        "real_val_loss": avg_real_val_loss,

        "train_seg_loss": train_loss_seg[-1],
        "val_seg_loss": val_loss_seg[-1],
        "train_depth_loss": train_loss_depth[-1],
        "val_depth_loss": val_loss_depth[-1],
        "train_wp_loss": train_loss_wp[-1],
        "val_wp_loss": val_loss_wp[-1],
        "train_speed_loss": train_loss_speed[-1],
        "val_speed_loss": val_loss_speed[-1],
        "train_brake_loss": train_loss_brake[-1],
        "val_brake_loss": val_loss_brake[-1],

        "train_seg_acc": avg_train_acc,
        "val_seg_acc": avg_val_acc,
        "train_depth_acc": avg_train_depth_acc,
        "val_depth_acc": avg_val_depth_acc
    })

    # Save best model
    # === Use real_val_loss as the main criterion ===
    if avg_real_val_loss < best_val_loss:
        best_val_loss = avg_real_val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved at epoch {best_epoch} with REAL val loss {best_val_loss:.4f}")
        
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     best_epoch = epoch + 1
    #     torch.save(model.state_dict(), best_model_path)
    #     print(f"New best model saved at epoch {best_epoch} with val loss {best_val_loss:.4f}")

print(f"\nTraining complete. Best model was from epoch {best_epoch} with val loss {best_val_loss:.4f}")
# ----------------------------
# Save Training Performance
# ----------------------------
stats_df = pd.DataFrame(training_stats)
stats_df.to_csv(training_csv_path, index=False)

# Plot Training vs Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig(loss_plot_path)
plt.show()

# Plot Semantic Segmentation Accuracy vs Epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_seg_acc, label="Train Seg Accuracy")
plt.plot(range(1, num_epochs + 1), val_seg_acc, label="Validation Seg Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Semantic Segmentation Accuracy vs Epochs")
plt.legend()
plt.savefig(seg_acc_plot_path)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_depth_acc, label="Train Depth Accuracy")
plt.plot(range(1, num_epochs + 1), val_depth_acc, label="Validation Depth Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Depth Classification Accuracy vs Epochs")
plt.legend()
plt.savefig(os.path.join(save_dir, "depth_accuracy_plot.png"))
plt.show()

stats_df = pd.DataFrame(training_stats)
plt.figure(figsize=(10, 5))
plt.plot(stats_df["epoch"], stats_df["real_train_loss"], label="Real Train Loss")
plt.plot(stats_df["epoch"], stats_df["real_val_loss"], label="Real Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Real (Unnormalized) Loss over Epochs")
plt.legend()
plt.savefig(os.path.join(save_dir, "real_loss_plot.png"))
plt.show()

def plot_loss_curve(train, val, title, filename):
    plt.figure(figsize=(10,5))
    plt.plot(range(1, num_epochs + 1), train, label='Train')
    plt.plot(range(1, num_epochs + 1), val, label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

plot_loss_curve(train_loss_seg, val_loss_seg, "Segmentation Loss", "seg_loss.png")
plot_loss_curve(train_loss_depth, val_loss_depth, "Depth Loss", "depth_loss.png")
plot_loss_curve(train_loss_wp, val_loss_wp, "Waypoint Loss", "wp_loss.png")
plot_loss_curve(train_loss_speed, val_loss_speed, "Speed Loss", "speed_loss.png")
plot_loss_curve(train_loss_brake, val_loss_brake, "Brake Loss", "brake_loss.png")

# ----------------------------
# Save the Trained Model
# ----------------------------
torch.save(model.state_dict(), final_model_path)
print("Model saved!")