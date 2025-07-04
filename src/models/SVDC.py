import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.decoder import DepthRegression
from src.models.encoder import HistogramEncoder
from src.models.decoder import UpSampleBN
from src.models.flow_warp import SPyNet,flow_warp
from src.models.second_order_deform import SecondOrderDeformableAlignment
from src.models.conv import ResidualBlocksWithInputConv

from src.models.dvsr import PixelShufflePack, get_pos_encoding
from src.models.penet_util import ENet

class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionExtractor, self).__init__()

        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.samconv(x)
        return self.sigmoid(x)


class SelectiveConvLayer(nn.Module):
    def __init__(self, input_dim=128, output_dim=256, small_kernel_size=1, large_kernel_size=3):
        super(SelectiveConvLayer, self).__init__()
        self.small_conv =  nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0)
        self.large_conv =  nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, att, x):
        h = self.small_conv(x) * att + self.large_conv(x) * (1 - att)
        return h


class Refine(nn.Module):
    def __init__(self,):
        super(Refine, self).__init__()
        self.layer0 = PixelShufflePack(128 + 16, 32, 2, upsample_kernel=3)  # [1,32,480,640]
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.layer = nn.Sequential(
                        self.lrelu,
                        nn.Conv2d(32, 16, 3, 1, 1),  # [1,64,480,640]
                        self.lrelu,
                        nn.Conv2d(16, 1, 3, 1, 1))  # [1,2,480,640]

    def forward(self, input):
        feat = self.layer0(input)
        out = self.layer(feat)
        return feat, out


class SoulpowerEncoder(nn.Module):
    def __init__(self):
        super(SoulpowerEncoder, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.adjust = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            self.lrelu,
            nn.Conv2d(16, 16, 3, 2, 1),
            self.lrelu,)
        self.conv0 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1),
            self.lrelu,)
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            self.lrelu,)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            self.lrelu,)

    def forward(self, x):
        features = [self.adjust(x)]
        features.append(self.conv0(features[-1]))
        features.append(self.conv1(features[-1]))
        features.append(self.conv2(features[-1]))

        return features[1:]


class Decoder(nn.Module):
    def __init__(self, num_classes=1):
        super(Decoder, self).__init__()
        encoder_channels = [512, 256, 128, 64, 32]
        decoder_channels = [256, 256, 128, 64, 32]
        self.conv4 = nn.Sequential(
            nn.Conv2d(encoder_channels[0], decoder_channels[0], kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_channels[0], decoder_channels[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.up1 = UpSampleBN(skip_input=decoder_channels[0] + encoder_channels[1], output_features=decoder_channels[1])
        self.up2 = UpSampleBN(skip_input=decoder_channels[1] + encoder_channels[2], output_features=decoder_channels[2])
        self.up3 = UpSampleBN(skip_input=decoder_channels[2] + encoder_channels[3], output_features=decoder_channels[3])
        self.up4 = UpSampleBN(skip_input=decoder_channels[3] + encoder_channels[4], output_features=decoder_channels[4])
        self.conv3 = nn.Sequential(
            nn.Conv2d(decoder_channels[1], decoder_channels[1], kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_channels[1], decoder_channels[2], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(decoder_channels[2], decoder_channels[2], kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_channels[2], decoder_channels[3], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_channels[3], decoder_channels[4], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(decoder_channels[4], decoder_channels[4], kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_channels[4], num_classes, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )


    def forward(self, img_features, hist_features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = img_features
        depth_feat1, depth_feat2, depth_feat3 = hist_features
        x_d4 = self.conv4(x_block4)                                  # [b, 256, 15, 20]
        x_d3 = self.up1(x_d4, x_block3)                              # [b, 256, 30, 40]
        x_d3 = self.conv3(x_d3)                                      # [b, 128, 30, 40]
        x_d3_fused = depth_feat3

        x_d3 = torch.cat([x_d3, x_d3_fused], dim=1)              # [b, 256, 30, 40]
        x_d2 = self.up2(x_d3, x_block2)                                 # [b, 128, 60, 80]
        x_d2 = self.conv2(x_d2)
        x_d2_fused = depth_feat2

        x_d2 = torch.cat([x_d2, x_d2_fused], dim=1)
        x_d1 = self.up3(x_d2, x_block1)
        x_d1 = self.conv1(x_d1)

        x_d1_fused = depth_feat1
        x_d1 = torch.cat([x_d1, x_d1_fused], dim=1)
        x_d0 = self.up4(x_d1, x_block0)
        out = self.conv0(x_d0)

        return out

class SVDC_3frame(nn.Module):
    def __init__(self, n_bins=100, min_val=0.1, max_val=10, norm='linear'):
        super(SVDC_3frame, self).__init__()
        self.cpu_cache = False
        self.is_mirror_extended = False
        self.mid_channels = 32
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.img_encoder = ENet()
        self.SpEncoder = SoulpowerEncoder()
        self.conf_pred_0 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
                                         self.lrelu)
        self.conf_pred_1 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                                         self.lrelu)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.refine = Refine()


        self.hist_encoder = HistogramEncoder()
        self.depth_head = DepthRegression(in_channels=32 * 3 * 2 + 2, dim_out=n_bins, norm=norm)
        self.decoder = Decoder(num_classes=32)
        self.conv_guide = nn.Sequential(
            nn.Conv2d(1 + 6, 16, kernel_size=3, stride=1, padding=1),
            self.LeakyReLU,
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            self.LeakyReLU,
            ResidualBlocksWithInputConv(16, 16, 1),
        )

        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        self.spynet_fix = SPyNet(pretrained="weights/spynet_20210409-c6c1bd09.pth")
        self.spynet = SPyNet(pretrained="weights/spynet_20210409-c6c1bd09.pth")

        mid_channels = 32
        max_residue_magnitude = 10
        num_blocks = 2

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.deform_align["hg_1"] = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.backbone["hg_1"] = nn.ModuleDict()

        # selective model
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(96)
        self.sconv = SelectiveConvLayer(96,96)

        modules = ["backward_1", "forward_1"]
        for i, module in enumerate(modules):
            self.deform_align["hg_1"][module] = SecondOrderDeformableAlignment(
                3,
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=8,
                max_residue_magnitude=max_residue_magnitude,
            )

            self.backbone["hg_1"][module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks
            )

        self._reset_parameters()

    def _reset_parameters(self):
        modules = [self.depth_head, self.decoder, self.conv_out, self.SpEncoder]
        for s in modules:
            for m in s.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def check_parameters(self,model):
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleDict):
                print(f"Parameters for ModuleDict '{name}':")
                for sub_name, sub_module in module.items():
                    self.check_parameters(sub_module)
            elif isinstance(module, nn.Module):
                print(f"Parameters for Module '{name}':")
                for param_name, param in module.named_parameters():
                    print(f"{param_name}: {param.requires_grad}")
            else:
                print(f"{name} is not a recognized module.")
    def compute_flow(self, guides,upscale = 2):
        """Compute optical flow using SPyNet for feature alignment.
        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.
        Args:
            guides (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            hg_idx: Identify processing stage: init stage or refine stage
        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = guides.size()
        guides_1 = guides[:, :-1, :, :, :]
        guides_2 = guides[:, 1:, :, :, :]

        guides_1 = guides_1.reshape(-1, c, h, w)
        guides_2 = guides_2.reshape(-1, c, h, w)


        flows_backward = self.spynet(guides_1, guides_2)
        flows_backward = F.interpolate(
                input = flows_backward,
                scale_factor=upscale,
                mode='bilinear',
                align_corners=True).view(n, t - 1, 2, h * upscale, w * upscale) * upscale


        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:

            guides_1 = guides_1.reshape(-1, c, h, w)
            guides_2 = guides_2.reshape(-1, c, h, w)
            flows_forward = self.spynet(guides_2, guides_1)
            flows_forward = F.interpolate(
                input=flows_forward,
                scale_factor=upscale,
                mode='bilinear',
                align_corners=True).view(n, t - 1, 2, h * upscale, w * upscale) * upscale

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name, hg_idx):
        """Propagate the latent features throughout the sequence.
        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h/4, w/4).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h/4, w/4).
            module_name (str): The name of the propagation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
            hg_idx: Identify processing stage: init stage or refine stage
        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size() ## 1/4 resolution of final output

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]

        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)  # [n,c,h/4,w/4]
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[f"hg_{hg_idx}"][module_name](
                    feat_prop, cond, flow_n1, flow_n2
                ) # fea_pop

                # concatenate and residual blocks

            feat = (
                [feat_current]
                + [feats[k][idx] for k in feats if k not in ["spatial", module_name]]
                + [feat_prop]
            )
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[f"hg_{hg_idx}"][module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def forward(self, input_data, **kwargs):
        x = input_data['rgb']  # [1, 3, 480, 640]   #[1,3,3,480,640]
        n, t, c, h, w = x.shape

        flow_backward = self.spynet_fix(x[:,1],x[:,2]) # now -> next
        flow_forward = self.spynet_fix(x[:,1],x[:,0])# now -> prev
        flow_1to3 = self.spynet_fix(x[:,2],x[:,0])

        additional_data = input_data['additional']
        sparse_depth = additional_data['sparse_depth'] # [B,t,c,h,w]
        position = additional_data['position']

        hist_features_0 = self.SpEncoder(sparse_depth[:, 0])
        hist_features_1 = self.SpEncoder(sparse_depth[:, 1])
        hist_features_2 = self.SpEncoder(sparse_depth[:, 2])

        img_features_0 = self.img_encoder(x[:, 0, :, :, :], sparse_depth[:,0], position, hist_features_0)
        img_features_1 = self.img_encoder(x[:, 1, :, :, :], sparse_depth[:,1], position, hist_features_1)
        img_features_2 = self.img_encoder(x[:, 2, :, :, :], sparse_depth[:,2], position, hist_features_2)

        unet_out_0 = self.decoder(img_features_0,hist_features_0)
        unet_out_1 = self.decoder(img_features_1, hist_features_1)
        unet_out_2 = self.decoder(img_features_2, hist_features_2)

        feats = {}
        feats["spatial"] = [unet_out_0,unet_out_1,unet_out_2]

        x_downsample_0 = F.interpolate(x[:,0].view(-1, c, int(h), int(w)),scale_factor=0.25,mode="bilinear",).view(n, c, int(h/4), int(w/4))
        x_downsample_1 = F.interpolate(x[:,1].view(-1, c, int(h), int(w)),scale_factor=0.25,mode="bilinear",).view(n, c, int(h/4), int(w/4))
        x_downsample_2 = F.interpolate(x[:,2].view(-1, c, int(h), int(w)),scale_factor=0.25,mode="bilinear",).view(n, c, int(h/4), int(w/4))
        x_downsample = torch.stack([x_downsample_0, x_downsample_1, x_downsample_2], dim = 1)
        flows_forward, flows_backward = self.compute_flow(x_downsample,upscale=2)

        # feature propagation
        for iter_ in [1]:
            for direction in ["backward", "forward"]:
                module = f"{direction}_{iter_}"

                feats[module] = []

                if direction == "backward":
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module, 1)

        hr_list = []
        for i in range(0, t):
            hr = [feats[k].pop(0) for k in feats if k != "spatial"]
            hr.insert(0, feats["spatial"][i])
            hr = torch.cat(hr, dim=1)
            hr_list.append(hr)
        hr_list = [self.cam(x) * x for x in hr_list]
        att = [self.sam(x) for x in hr_list]
        hr_list = [self.sconv(att_i,x_i) for att_i,x_i in zip(att,hr_list)]

        # recurruent fusion
        flows_0 = torch.zeros_like(flows_forward[:,0])
        feat_0 = torch.zeros_like(hr_list[0])

        dec_fusion_0 = torch.cat([hr_list[0],feat_0,flows_0],dim = 1)
        dec_fusion_1 = torch.cat([hr_list[1],hr_list[0],flows_forward[:,0]],dim = 1)
        dec_fusion_2 = torch.cat([hr_list[2], hr_list[1], flows_forward[:,1]], dim = 1)

        bin_widths_normed_0, range_attention_maps_0 = self.depth_head(dec_fusion_0)  # [b, n_bins]
        bin_widths_normed_1, range_attention_maps_1 = self.depth_head(dec_fusion_1)
        bin_widths_normed_2, range_attention_maps_2 = self.depth_head(dec_fusion_2)

        out_0 = self.conv_out(range_attention_maps_0)
        out_1 = self.conv_out(range_attention_maps_1)
        out_2 = self.conv_out(range_attention_maps_2)

        bin_widths_0 = (self.max_val - self.min_val) * bin_widths_normed_0
        bin_widths_0 = nn.functional.pad(bin_widths_0, (1, 0), mode='constant', value=self.min_val)
        bin_widths_1 = (self.max_val - self.min_val) * bin_widths_normed_1
        bin_widths_1 = nn.functional.pad(bin_widths_1, (1, 0), mode='constant', value=self.min_val)
        bin_widths_2 = (self.max_val - self.min_val) * bin_widths_normed_2
        bin_widths_2 = nn.functional.pad(bin_widths_2, (1, 0), mode='constant', value=self.min_val)

        bin_edges_0 = torch.cumsum(bin_widths_0, dim=1)
        bin_edges_1 = torch.cumsum(bin_widths_1, dim=1)
        bin_edges_2 = torch.cumsum(bin_widths_2, dim=1)

        centers_0 = 0.5 * (bin_edges_0[:, :-1] + bin_edges_0[:, 1:])
        n, dout = centers_0.size()
        centers_0 = centers_0.view(n, dout, 1, 1)

        centers_1 = 0.5 * (bin_edges_1[:, :-1] + bin_edges_1[:, 1:])
        n, dout = centers_1.size()
        centers_1 = centers_1.view(n, dout, 1, 1)

        centers_2 = 0.5 * (bin_edges_2[:, :-1] + bin_edges_2[:, 1:])
        n, dout = centers_2.size()
        centers_2 = centers_2.view(n, dout, 1, 1)
        bin_edges = torch.stack([bin_edges_0,bin_edges_1,bin_edges_2],dim = 1)

        pitch = 16
        pred_0 = torch.sum(out_0 * centers_0, dim=1, keepdim=True)
        pred_1 = torch.sum(out_1 * centers_1, dim=1, keepdim=True)
        pred_2 = torch.sum(out_2 * centers_2, dim=1, keepdim=True)

        conf_0 = self.conf_pred_0(range_attention_maps_0)
        conf_1 = self.conf_pred_0(range_attention_maps_1)
        conf_2 = self.conf_pred_0(range_attention_maps_2)

        B, _, H, W = x[:,0].shape

        conf_0 = nn.functional.interpolate(conf_0, x.shape[-2:], mode='bilinear',
                                           align_corners=True)
        conf_1 = nn.functional.interpolate(conf_1, x.shape[-2:], mode='bilinear',
                                           align_corners=True)
        conf_2 = nn.functional.interpolate(conf_2, x.shape[-2:], mode='bilinear',
                                           align_corners=True)

        pred_0 = nn.functional.interpolate(pred_0, x.shape[-2:], mode='bilinear',
                                           align_corners=True)
        pred_1 = nn.functional.interpolate(pred_1, x.shape[-2:], mode='bilinear',
                                           align_corners=True)
        pred_2 = nn.functional.interpolate(pred_2, x.shape[-2:], mode='bilinear',
                                           align_corners=True)

        pos_enc = get_pos_encoding(B, H, W, pitch).to(x.device)

        feat_0 = self.conv_guide(torch.cat([pred_0, pos_enc], dim=1))
        feat_1 = self.conv_guide(torch.cat([pred_1, pos_enc], dim=1))
        feat_2 = self.conv_guide(torch.cat([pred_2, pos_enc], dim=1))

        refine_feat_conf_0, refine_depth_0 = self.refine(torch.cat([feat_0,range_attention_maps_0], dim=1))
        refine_depth_0 = torch.relu(refine_depth_0)
        refine_conf_0 = self.conf_pred_1(refine_feat_conf_0)
        weight = nn.functional.softmax(torch.cat((refine_conf_0, conf_0), dim=1), dim=1)
        refine_conf_0, conf_0 = torch.chunk(weight, 2, dim=1)
        final_0 = refine_depth_0 * refine_conf_0 + pred_0 * conf_0

        refine_feat_conf_1, refine_depth_1 = self.refine(torch.cat([feat_1,range_attention_maps_1], dim=1))
        refine_depth_1 = torch.relu(refine_depth_1)
        refine_conf_1 = self.conf_pred_1(refine_feat_conf_1)
        weight = nn.functional.softmax(torch.cat((refine_conf_1, conf_1), dim=1), dim=1)
        refine_conf_1, conf_1 = torch.chunk(weight, 2, dim=1)
        final_1 = refine_depth_1 * refine_conf_1 + pred_1 * conf_1

        refine_feat_conf_2, refine_depth_2 = self.refine(torch.cat([feat_2,range_attention_maps_2], dim=1))
        refine_depth_2 = torch.relu(refine_depth_2)
        refine_conf_2 = self.conf_pred_1(refine_feat_conf_2)
        weight = nn.functional.softmax(torch.cat((refine_conf_2, conf_2), dim=1), dim=1)
        refine_conf_2, conf_2 = torch.chunk(weight, 2, dim=1)
        final_2 = refine_depth_2 * refine_conf_2 + pred_2 * conf_2

        pred_lowres = torch.stack([pred_0,pred_1,pred_2],dim = 1)
        pred = torch.stack([final_0,final_1,final_2],dim = 1)

        return bin_edges, pred, flow_forward, flow_backward,flow_1to3,pred_lowres

    def get_1x_lr_params(self):  # lr/10 learning rate
        modules = [self.spynet]
        for m in modules:
            yield from m.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.depth_head, self.conv_out,self.conv_guide, self.refine, self.conf_pred_1, self.conf_pred_0, self.decoder, self.SpEncoder,self.deform_align, self.backbone,self.img_encoder,self.cam,self.sam,self.sconv]
        for m in modules:
            yield from m.parameters()

    def get_unfrozen_params(self):
        modules = [self.conv_guide, self.refine, self.conf_pred_1, self.conf_pred_0]
        for m in modules:
            yield from m.parameters()






