import torch
import numpy as np
import math
from torch.nn import functional as F
from typing import Optional


def find_maxima(x):
    col_max = torch.amax(x, axis=1)
    row_max = torch.amax(x, axis=2)

    cols = torch.argmax(input=col_max, dim=1).to(torch.float32)
    rows = torch.argmax(input=row_max, dim=1).to(torch.float32)
    cols = cols.reshape((-1, 1))
    rows = rows.reshape((-1, 1))

    maxima = torch.cat([rows, cols], -1)

    return maxima


def fix(x):
    x = torch.where(x >= 0, torch.floor(x), torch.ceil(x))
    return x


def largest_factor(x):
    n = n_downsample(x)
    return x // 2 ** n


def n_downsample(x):
    n = 0
    while x % 2 == 0 and x > 2:
        n += 1
        x /= 2.0
    return n


def _col_kernel(upsampled_region_size, upsample_factor, axis_offsets, data_shape):

    data_shape_float = data_shape.to(torch.float32)
    col_constant = (data_shape_float[2] * upsample_factor).to(torch.complex64)
    col_constant = -1j * 2 * np.pi / col_constant

    col_kernel_a = torch.arange(
        0, data_shape_float[2], device=data_shape.device, dtype=torch.float32
    )
    col_kernel_a = torch.fft.fftshift(col_kernel_a)
    col_kernel_a = torch.reshape(col_kernel_a, (-1, 1))
    col_kernel_a -= torch.floor(data_shape_float[2] / 2.0)
    col_kernel_a = torch.reshape(col_kernel_a, (1, -1))
    col_kernel_a = col_kernel_a.repeat(data_shape[0], 1)

    col_kernel_b = torch.arange(
        0, upsampled_region_size, device=data_shape.device, dtype=torch.float32
    )
    col_kernel_b = torch.reshape(col_kernel_b, (1, -1))
    col_kernel_b = col_kernel_b.repeat(data_shape[0], 1)
    col_kernel_b = torch.transpose(col_kernel_b, 0, 1)
    col_kernel_b -= torch.transpose(
        axis_offsets[:, 1].unsqueeze(-1), 0, 1
    )  # double check the unsqueezing so the transpose dims work out
    col_kernel_b = torch.transpose(col_kernel_b, 0, 1)

    col_kernel_a = torch.unsqueeze(col_kernel_a, 1)
    col_kernel_b = torch.unsqueeze(col_kernel_b, -1)

    col_kernel = col_kernel_a * col_kernel_b
    col_kernel = col_kernel.permute((0, 2, 1))
    col_kernel = col_constant * col_kernel.to(torch.complex64)
    col_kernel = torch.exp(col_kernel)
    return col_kernel


def _row_kernel(upsampled_region_size, upsample_factor, axis_offsets, data_shape):

    data_shape_float = data_shape.to(torch.float32)
    row_constant = (data_shape_float[1] * upsample_factor).to(torch.complex64)
    row_constant = -1j * 2 * np.pi / row_constant

    row_kernel_a = torch.arange(
        0, upsampled_region_size, device=data_shape.device, dtype=torch.float32
    )
    row_kernel_a = torch.reshape(row_kernel_a, (1, -1))
    row_kernel_a = row_kernel_a.repeat(data_shape[0], 1)
    row_kernel_a = torch.transpose(row_kernel_a, 0, 1)
    row_kernel_a = row_kernel_a - axis_offsets[:, 0]

    row_kernel_b = torch.arange(
        0, data_shape_float[1], device=data_shape.device, dtype=torch.float32
    )
    row_kernel_b = torch.fft.fftshift(row_kernel_b)
    row_kernel_b = torch.reshape(row_kernel_b, (1, -1))
    row_kernel_b = row_kernel_b.repeat(data_shape[0], 1)
    row_kernel_b = row_kernel_b - torch.floor(data_shape_float[1] / 2.0)

    row_kernel_a = torch.unsqueeze(row_kernel_a, 1)
    row_kernel_b = torch.unsqueeze(row_kernel_b, -1)

    row_kernel = row_kernel_a.permute((2, 1, 0)) * row_kernel_b
    row_kernel = row_kernel.permute((0, 2, 1))
    row_kernel = row_constant * row_kernel.to(torch.complex64)

    row_kernel = torch.exp(row_kernel)

    return row_kernel


def _upsampled_dft(data, upsampled_region_size, upsample_factor, axis_offsets):
    data_shape = torch.tensor(data.shape, device=data.device)

    col_kernel = _col_kernel(
        upsampled_region_size, upsample_factor, axis_offsets, data_shape
    )
    row_kernel = _row_kernel(
        upsampled_region_size, upsample_factor, axis_offsets, data_shape
    )

    upsampled_dft = (row_kernel @ data) @ col_kernel

    return upsampled_dft


# tensorflow source: https://github.com/jgraving/DeepPoseKit/blob/cecdb0c8c364ea049a3b705275ae71a2f366d4da/deepposekit/models/backend/backend.py#L176
def find_subpixel_maxima(
    heatmaps, kernel_size, sigma, upsample_factor, coordinate_scale, confidence_scale
):  # data format implictly channels_first
    map_shape = heatmaps.shape
    batch = map_shape[0]
    channels = map_shape[1]
    row = map_shape[2]
    col = map_shape[3]
    heatmaps = heatmaps.reshape(shape=(batch * channels, row, col))  # check data types

    # dpk_kernel = dpk.models.backend.utils.gaussian_kernel_2d(kernel_size.cpu(), sigma.cpu())
    size = kernel_size
    x = torch.arange(
        -(size // 2), (size // 2) + 1, dtype=torch.float32, device=heatmaps.device
    )
    kernel = torch.tensor(1 / (sigma * math.sqrt(2 * np.pi)), device=heatmaps.device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2) * kernel  # could be fishy
    kernel = torch.unsqueeze(kernel, -1)
    kernel = kernel @ torch.transpose(kernel, 0, 1)  # check dims

    kernel = torch.unsqueeze(kernel, 0)
    max_vals = torch.amax(heatmaps.reshape(shape=(-1, row * col)), dim=1)
    max_vals = max_vals.reshape(shape=(-1, 1)) / confidence_scale
    row_pad = row // 2 - kernel_size // 2
    col_pad = col // 2 - kernel_size // 2
    padding = (
        col_pad,
        col_pad - 1,
        row_pad,
        row_pad - 1,
    )  # padding order goes last dim, second to last dim, ...
    kernel = F.pad(kernel, padding)
    row_center = row_pad + (kernel_size // 2)
    col_center = col_pad + (kernel_size // 2)
    center = torch.stack([torch.tensor(row_center), torch.tensor(col_center)])
    center = torch.unsqueeze(center, 0)
    center = center.to(torch.float32)
    # dpk_registration = dpk.models.backend.registration._upsampled_registration(heatmaps.cpu(), kernel.cpu(), upsample_factor.cpu())

    target_shape = heatmaps.shape
    target_image = heatmaps.reshape(target_shape[:3])
    src_shape = kernel.shape
    kernel = kernel.reshape(src_shape[:3])

    src_image = kernel.to(torch.complex64)
    target_image = heatmaps.to(torch.complex64)
    src_freq = torch.fft.fft2(src_image)

    target_freq = torch.fft.fft2(target_image)

    shape = torch.tensor(src_freq.shape[1:3], device=heatmaps.device)
    shape = shape.reshape(shape=(1, 2))
    shape = shape.to(torch.float32)
    shape = shape.repeat(target_freq.shape[0], 1)  # check repeat vs tile
    image_product = src_freq * torch.conj(target_freq)
    cross_correlation = torch.fft.ifft2(image_product)

    maxima = find_maxima(torch.abs(cross_correlation))
    midpoints = fix(shape.to(torch.float32) / 2)

    shifts = maxima
    shifts = torch.where(shifts > midpoints, shifts - shape, shifts)
    shifts = torch.round(shifts * upsample_factor) / upsample_factor

    upsampled_region_size = torch.ceil(upsample_factor * 1.5)
    dftshift = fix(upsampled_region_size / 2.0)
    normalization = torch.tensor(
        torch.numel(src_freq[0]), device=heatmaps.device, dtype=torch.float32
    )
    normalization *= upsample_factor ** 2
    sample_region_offset = dftshift - shifts * upsample_factor

    data = torch.conj(image_product)

    upsampled_dft = _upsampled_dft(
        data, upsampled_region_size, upsample_factor, sample_region_offset
    )
    cross_correlation = torch.conj(upsampled_dft)
    cross_correlation /= normalization.to(torch.complex64)
    cross_correlation = torch.abs(cross_correlation)

    maxima = find_maxima(cross_correlation)
    maxima = maxima - dftshift
    shifts = shifts + maxima / upsample_factor

    shifts = center - shifts
    shifts *= coordinate_scale
    shifts = torch.flip(shifts, [-1])  # CHECK BEHAVIOR IS SAME AS TENSORFLOW INDEXING
    maxima = torch.cat([shifts, max_vals], -1)

    maxima = maxima.reshape((batch, channels, 3))
    return maxima


class SubPixelMaxima:  # Add tensor typing
    def __init__(
        self,
        output_shape: tuple,
        output_sigma: torch.Tensor,
        upsample_factor: torch.Tensor,
        coordinate_scale: torch.Tensor,
        confidence_scale: torch.Tensor,
        threshold: float = None,
        device: str = "cpu",
    ):

        self.output_shape = output_shape
        self.output_sigma = output_sigma
        self.upsample_factor = upsample_factor
        self.coordinate_scale = coordinate_scale
        self.confidence_scale = confidence_scale
        self.threshold = threshold
        self.device = device

    @property
    def kernel_size(self):
        kernel_size = np.min(self.output_shape)
        kernel_size = (kernel_size // largest_factor(kernel_size)) + 1
        return torch.tensor(kernel_size, device=self.device)

    def run( #TODO: maybe we should see if we can add batch functionality
        self,
        heatmaps_1: torch.Tensor,
        heatmaps_2: torch.Tensor=None,  # Enables the function to be run with only one set of keypoints
    ):
        keypoints_1 = find_subpixel_maxima(
            heatmaps_1,  # .detach(),  TODO: is there a reason to detach?
            self.kernel_size,
            self.output_sigma,
            self.upsample_factor,
            self.coordinate_scale,
            self.confidence_scale,
        )

        if heatmaps_2 == None:
            return self.use_threshold(keypoints_1)

        keypoints_2 = find_subpixel_maxima(
            heatmaps_2,  #.detach(),
            self.kernel_size,
            self.output_sigma,
            self.upsample_factor,
            self.coordinate_scale,
            self.confidence_scale,
        )
        return self.use_threshold(keypoints_1), self.use_threshold(keypoints_2)

    def use_threshold(self, keypoints: torch.tensor):  # TODO: figure out what to do with batched masking, different elements of the batch could be masked into different sizes, which would cause a shape mismatch, could just turn masked elements into nans
        if not self.threshold:
            num_threshold = torch.tensor(-1, device=self.device)
        else:
            num_threshold = torch.tensor(self.threshold, device=keypoints.device)
        #print(keypoints.shape)
        batch_dim, num_bodyparts, _ = keypoints.shape
        mask = torch.gt(keypoints[:, :, 2], num_threshold)
        #print(mask.shape)
        mask = mask.unsqueeze(-1)
        keypoints = torch.masked_select(keypoints, mask).reshape(
            batch_dim, -1, 3
        )
        #print(keypoints.shape)
        confidence = keypoints[:, :, 2]
        keypoints = keypoints[:, :, :2]
        return keypoints.reshape(keypoints.shape[0], -1), confidence


def format_mouse_data(data_arr):
    # TODO: assume that data is a csv file or pandas dataframe
    # with first line indicating view and second line indicating body part names
    # access the the unique elements of the view column instead of :7 and 8:15
    data_arr_top = data_arr[:, :7, :]  # mouse data info hardcoded here
    data_arr_bot = data_arr[:, 8:15, :]
    data_arr_top = data_arr_top.permute(2, 0, 1).reshape(2, -1)
    data_arr_bot = data_arr_bot.permute(2, 0, 1).reshape(2, -1)
    data_arr = torch.cat([data_arr_top, data_arr_bot], dim=0)
    return data_arr
