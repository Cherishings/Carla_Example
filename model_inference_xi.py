# model_inference.py

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

NUM_CLASSES = 4  # Background, Road, Cars, Lane Marks

class VehicleNNController(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(VehicleNNController, self).__init__()

        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )

        self.encoder = self.unet.encoder
        self.decoder = self.unet.decoder
        self.segmentation_head = self.unet.segmentation_head

        self.flatten = nn.Flatten()
        self.fc_backbone = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        self.meta_input_net = nn.Sequential(
            nn.BatchNorm1d(3),
            nn.Linear(3, 20),
            nn.ReLU(),
            nn.Linear(20, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU()
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(128 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.output_waypoints = nn.Linear(32, 10)
        self.output_speed = nn.Linear(32, 1)
        self.output_brake = nn.Linear(32, 1)

    def forward(self, x_img, target_point, current_speed):
        features = self.encoder(x_img)
        decoder_output = self.decoder(*features)
        semantic_out = self.segmentation_head(decoder_output)

        x_deep = features[-1]
        flat_feat = self.flatten(x_deep)
        features_128 = self.fc_backbone(flat_feat)

        meta = torch.cat([target_point, current_speed.unsqueeze(1)], dim=1)
        meta_feat = self.meta_input_net(meta)

        combined = torch.cat([features_128, meta_feat], dim=1)
        fused = self.fc_combined(combined)

        waypoints = self.output_waypoints(fused)
        speed = self.output_speed(fused)
        brake = torch.sigmoid(self.output_brake(fused))

        return semantic_out, waypoints, speed, brake
