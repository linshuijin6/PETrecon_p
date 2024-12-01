import numpy as np
import torch
# from utils import fourier_filter, get_pad_width

import torch.nn.functional as F
from matplotlib import pyplot as plt

from model.network_swinTrans import SwinIR


def fourier_filter(name, size, device="cpu"):
    n = torch.cat((torch.arange(1, size / 2 + 1, 2, dtype=int), torch.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = torch.zeros(size)
    f[0] = 0.25
    pi = torch.deg2rad(torch.tensor(180.0))
    f[1::2] = -1 / (pi * n) ** 2
    fourier_filter = 2 * torch.real(torch.fft.fft(f))

    if name == "ramp":
        pass
    elif name == "shepp_logan":
        omega = torch.pi * torch.fft.fftfreq(size)[1:]
        fourier_filter[1:] *= torch.sin(omega) / omega

    elif name == "cosine":
        freq = torch.linspace(0, torch.pi - (torch.pi / size), size)
        cosine_filter = torch.fft.fftshift(torch.sin(freq))
        fourier_filter *= cosine_filter

    elif name == "hamming":
        fourier_filter *= torch.fft.fftshift(torch.hamming_window(size, periodic=False))

    elif name == "hann":
        fourier_filter *= torch.fft.fftshift(torch.hann_window(size, periodic=False))
    else:
        print(f"[TorchRadon] Error, unknown filter type '{name}', available filters are: 'ramp', 'shepp_logan', 'cosine', 'hamming', 'hann'")

    filter = fourier_filter.to(device)

    return filter


def get_pad_width(image_size):
    """
    Pads the input image to make it square and centered, with non-zero elements in the middle.

    Args:
        image (torch.Tensor): Input image tensor of shape (batch_size, 1, W, W)

    Returns:
        padded_image (torch.Tensor): Padded image tensor
        pad_width (list): Amount of padding applied to each dimension
    """

    # Compute diagonal and padding sizes
    diagonal = (2**0.5) * image_size
    pad = [int(torch.ceil(torch.tensor(diagonal - s))) for s in (image_size, image_size)]

    # Compute new and old centers
    new_center = [(s + p) // 2 for s, p in zip((image_size, image_size), pad)]
    old_center = [s // 2 for s in (image_size, image_size)]

    # Compute padding before and after
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = tuple((pb, p - pb) for pb, p in zip(pad_before, pad))

    return pad_width


class Radon(torch.nn.Module):
    """
    Radon Transformation

    Args:
        thetas (int): list of angles for radon transformation (default: [0, np.pi])
        circle (bool): if True, only the circle is reconstructed (default: False)
        filter_name (str): filter for backprojection, can be "ramp" or "shepp_logan" or "cosine" or "hamming" or "hann" (default: "ramp")
        device: (str): device can be either "cuda" or "cpu" (default: cuda")
    """

    def __init__(self, n_theta=180, circle=False, filter_name="ramp", device="cuda"):
        super(Radon, self).__init__()
        self.n_angles = n_theta
        self.circle = circle
        self.filter_name = filter_name
        self.device = device

        # get angles
        thetas = torch.linspace(0, 180, self.n_angles, device=device)

        self.thetas = torch.deg2rad(thetas[:, None, None]).float().to(device)
        # self.cos_al, self.sin_al = thetas.cos(), thetas.sin()

    def forward(self, image):
        """Apply radon transformation on input image.

        Args:
            image (torch.tensor, (bzs, 1, W, H)): input image

        Returns:
            out (torch.tensor, (bzs, 1, W, angles)): sinogram
        """
        batch_size, _, image_size, _ = image.shape
        cos_angles, sin_angles = self.thetas.cos(), self.thetas.sin()
        # code for circle case
        if not self.circle:
            pad_width = get_pad_width(image_size)
            image_size = int(torch.ceil(torch.tensor((2 ** 0.5) * image_size)))
            new_img = F.pad(
                image,
                pad=[pad_width[1][0], pad_width[1][1], pad_width[0][0], pad_width[0][1]],
                mode="constant",
                value=0,
            )
        else:
            new_img = image

        # # Calculate rotated images
        rotated_images = []
        for cos_al, sin_al in zip(cos_angles, sin_angles):
            theta = (
                torch.tensor([[cos_al, sin_al, 0], [-sin_al, cos_al, 0]], dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            grid = F.affine_grid(theta, torch.Size([1, 1, image_size, image_size]), align_corners=True).to(self.device)
            rotated_img = F.grid_sample(new_img, grid.repeat(batch_size, 1, 1, 1), align_corners=True)

            rotated_images.append(rotated_img.sum(2))

        out_fl = torch.stack(rotated_images, dim=2)
        out_fl = out_fl.permute(0, 1, 3, 2)
        return out_fl

    def filter_backprojection(self, sinogram):
        """Apply (filtered) backprojection on sinogram.

        Args:
            input (torch.tensor, (bzs, 1, W, angles)): sinogram

        Returns:
            out (torch.tensor, (bzs, 1, W, H)): reconstructed image
        """

        bsz, _, det_count, _ = sinogram.shape
        cos_angles, sin_angles = self.thetas.cos(), self.thetas.sin()
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, det_count), torch.linspace(-1, 1, det_count)
        )
        grid_y = grid_y.to(self.device)
        grid_x = grid_x.to(self.device)
        reconstruction_circle = (grid_x ** 2 + grid_y ** 2) <= 1
        reconstructed_circle = reconstruction_circle.repeat(bsz, 1, 1, 1).to(self.device)

        projection_size_padded = max(64, int(2 ** (2 * torch.tensor(det_count)).float().log2().ceil()))
        self.pad_width_sino = projection_size_padded - det_count

        if self.filter_name is not None:
            filter = fourier_filter(name=self.filter_name, size=projection_size_padded, device=self.device)
            # Pad input
            padded_input = torch.nn.functional.pad(sinogram, [0, 0, 0, self.pad_width_sino], mode="constant", value=0)
            # Apply filter
            projection = torch.fft.fft(padded_input, dim=2) * filter[:, None]
            radon_filtered = torch.real(torch.fft.ifft(projection, dim=2))[:, :, :det_count, :]
        else:
            radon_filtered = sinogram

        reconstructed = torch.zeros((bsz, 1, det_count, det_count), device=self.device)
        y_grid = torch.linspace(-1, 1, self.n_angles).to(self.device)

        # Reconstruct using a for loop
        for i, (cos_al, sin_al, y_colume) in enumerate(zip(cos_angles, sin_angles, y_grid)):
            tgrid = (grid_x * cos_al - grid_y * sin_al).unsqueeze(0).unsqueeze(-1)
            y = torch.ones_like(tgrid) * y_colume
            grid = torch.cat((y, tgrid), dim=-1).to(self.device)
            # Apply grid_sample to the current angle
            radon_filtered = radon_filtered.to(self.device).float()
            rotated_img = F.grid_sample(
                radon_filtered, grid.repeat(bsz, 1, 1, 1), mode="bilinear", padding_mode="zeros", align_corners=True
            )
            rotated_img = rotated_img.view(bsz, 1, det_count, det_count)
            # Sum the rotated images for backprojection
            reconstructed += rotated_img

        # Circle
        pi = torch.deg2rad(torch.tensor(180.0))
        reconstructed[reconstructed_circle == 0] = 0.0
        reconstructed = reconstructed * pi / (2 * self.n_angles)

        if self.circle == False:
            # center crop reconstructed to the output size = det_count / (2**0.5)
            output_size = int(torch.floor(torch.tensor(det_count / (2 ** 0.5))))
            start_idx = (det_count - output_size) // 2
            end_idx = start_idx + output_size
            reconstructed = reconstructed[:, :, start_idx:end_idx, start_idx:end_idx]

        return reconstructed


if __name__ == '__main__':
    device = 'cuda:1'
    me_radon = Radon(n_theta=180, circle=True, device=device)
    data_n = np.load('./simulation_angular/angular_180/test_transverse_sinoLD.npy', allow_pickle=True)
    data_1n = np.load('./simulation_angular/angular_180/test_transverse_picHD.npy', allow_pickle=True)
    with torch.no_grad():
        sinoLD = torch.from_numpy(data_n).float().squeeze().unsqueeze(1).to(device)[:3, :, :, :]
        picHD = torch.from_numpy(data_1n).float().squeeze().unsqueeze(1).to(device)[:3, :, :, :]
        denoise_model_pre = SwinIR(upscale=1,
                                   in_chans=1,
                                   img_size=[128, 180],
                                   window_size=4,
                                   patch_size=[1, 45],
                                   img_range=1.0,
                                   depths=[2, 6, 2],
                                   embed_dim=180,
                                   num_heads=[3, 6, 12],
                                   mlp_ratio=2.0,
                                   upsampler='',
                                   resi_connection='1conv', ).to(device)
        denoise_model_pre.load_state_dict(torch.load('/home/ssddata/linshuijin/PETrecon/model/denoise_pre_weight_best.pth'))
        out_sino1 = denoise_model_pre(sinoLD)
        pic_1 = me_radon.filter_backprojection(sinoLD)
        out_sino2 = denoise_model_pre(out_sino1)
        pic_2 = me_radon.filter_backprojection(out_sino2)

    plt.imshow(pic_1[0, 0, :, :].cpu().numpy()), plt.show()
    plt.imshow(pic_2[0, 0, :, :].cpu().numpy()), plt.show()
    plt.imshow(picHD[0, 0, :, :].cpu().numpy()), plt.show()


