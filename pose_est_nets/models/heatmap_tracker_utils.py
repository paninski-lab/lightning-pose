def find_maxima(x):
    col_max = torch.amax(x, axis=1)
    row_max = torch.amax(x, axis=2)

    cols = torch.argmax(input = col_max, dim = 1).to(torch.float32)
    rows = torch.argmax(input = row_max, dim = 1).to(torch.float32)
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

#not used for now
def fftshift1d(x, axis=0):

    x_shape = torch.shape(x)
    x = torch.reshape(x, (-1, 1))
    n_samples = torch.cast(torch.shape(x)[0], torch.float32)
    even = n_samples / 2.0
    even = torch.round(even)
    even = even * 2.0
    even = torch.equal(n_samples, even)

    def true_fn():
        return x

    def false_fn():
        x_padded = torch.concat([x, torch.zeros((1, 1))], axis=0)
        return x_padded

    x = torch.cond(even, true_fn, false_fn)
    x1, x2 = torch.split(x, 2, axis=axis)

    def true_fn():
        return x2

    def false_fn():
        x2_unpadded = x2[:-1]
        return x2_unpadded

    x2 = torch.cond(even, true_fn, false_fn)
    x = torch.concat((x2, x1), axis=axis)
    x = torch.reshape(x, x_shape)

    return x

def _col_kernel(upsampled_region_size, upsample_factor, axis_offsets, data_shape):

    data_shape_float = data_shape.to(torch.float32)
    col_constant = (data_shape_float[2] * upsample_factor).to(torch.complex64)
    col_constant = -1j * 2 * np.pi / col_constant

    col_kernel_a = torch.range(0, data_shape_float[2], dtype=torch.float32)
    col_kernel_a = torch.fft.fftshift(col_kernel_a)
    col_kernel_a = torch.reshape(col_kernel_a, (-1, 1))
    col_kernel_a -= torch.floor(data_shape_float[2] / 2.0)
    col_kernel_a = torch.reshape(col_kernel_a, (1, -1))
    col_kernel_a = torch.tile(col_kernel_a, (data_shape[0], 1))

    col_kernel_b = torch.range(0, upsampled_region_size, dtype=torch.float32)
    col_kernel_b = torch.reshape(col_kernel_b, (1, -1))
    col_kernel_b = torch.tile(col_kernel_b, (data_shape[0], 1))
    col_kernel_b = torch.transpose(col_kernel_b)
    col_kernel_b -= torch.transpose(axis_offsets[:, 1])
    col_kernel_b = torch.transpose(col_kernel_b)

    col_kernel_a = torch.unsqueeze(col_kernel_a, 1)
    col_kernel_b = torch.unsqueeze(col_kernel_b, -1)

    col_kernel = col_kernel_a * col_kernel_b
    col_kernel = col_kernel.permute((0, 2, 1))
    col_kernel = col_constant * col_kernel.to(torch.complex64)
    col_kernel = torch.exp(col_kernel)
    return col_kernel


def _row_kernel(upsampled_region_size, upsample_factor, axis_offsets, data_shape):

    data_shape_float = torch.cast(data_shape, torch.float32)
    row_constant = torch.cast(data_shape_float[1] * upsample_factor, torch.complex64)
    row_constant = -1j * 2 * np.pi / row_constant

    row_kernel_a = torch.range(0, upsampled_region_size, dtype=torch.float32)
    row_kernel_a = torch.reshape(row_kernel_a, (1, -1))
    row_kernel_a = torch.tile(row_kernel_a, (data_shape[0], 1))
    row_kernel_a = torch.transpose(row_kernel_a)
    row_kernel_a = row_kernel_a - axis_offsets[:, 0]

    row_kernel_b = torch.range(0, data_shape_float[1], dtype=torch.float32)
    row_kernel_b = torch.fft.fftshift(row_kernel_b)
    row_kernel_b = torch.reshape(row_kernel_b, (1, -1))
    row_kernel_b = torch.tile(row_kernel_b, (data_shape[0], 1))
    row_kernel_b = row_kernel_b - torch.floor(data_shape_float[1] / 2.0)

    row_kernel_a = torch.unsqueeze(row_kernel_a, 1)
    row_kernel_b = torch.unsqueeze(row_kernel_b, -1)

    row_kernel = torch.transpose(row_kernel_a) * row_kernel_b
    row_kernel = row_kernel.permute(perm=(0, 2, 1))
    row_kernel = row_constant * row_kernel.to(torch.complex64)

    row_kernel = torch.exp(row_kernel)

    return row_kernel


def _upsampled_dft(data, upsampled_region_size, upsample_factor, axis_offsets):
    data_shape = data.shape

    col_kernel = _col_kernel(
        upsampled_region_size, upsample_factor, axis_offsets, data_shape
    )
    row_kernel = _row_kernel(
        upsampled_region_size, upsample_factor, axis_offsets, data_shape
    )

    upsampled_dft = ((row_kernel @ data) @ col_kernel)

    return upsampled_dft

#tensorflow source: https://github.com/jgraving/DeepPoseKit/blob/cecdb0c8c364ea049a3b705275ae71a2f366d4da/deepposekit/models/backend/backend.py#L176
def find_subpixel_maxima(self, heatmaps, kernel_size, sigma, upsample_factor, coordinate_scale, confidence_scale): #data format implictly channels_first
    map_shape = heatmaps.shape
    batch = map_shape[0]
    channels = map_shape[1]
    row = map_shape[2]
    col = map_shape[3]
    heatmaps = heatmaps.reshape(shape = (batch * channels, row, col)) #check data types
    x = torch.range( -(kernel_size // 2), (kernel_size // 2) + 1, dtype = torch.float32, device = 'cuda')
    kernel = 1 / (sigma * torch.sqrt(2 * np.pi))
    kernel *= torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = torch.unsqueeze(kernel, -1)
    kernel = kernel @ torch.transpose(kernel)
    kernel = torch.unsqueeze(kernel, 0)
    max_vals = torch.amax(heatmaps.reshape(shape = (-1, rows * cols)))
    max_vals = max_vals.reshape(shape = (-1, 1)) / confidence_scale
    row_pad = rows // 2 - kernel_size // 2
    col_pad = cols // 2 - kernel_size // 2
    padding = (row_pad, row_pad - 1, col_pad, col_pad - 1)
    kernel = F.pad(kernel, padding)
    row_center = row_pad + (kernel_size // 2)
    col_center = col_pad + (kernel_size // 2)
    center = torch.stack([row_center, col_center])
    center = torch.unsqueeze(center, 0)
    center.dtype = torch.float32
    target_image = heatmaps.reshape(map_shape[:3])
    src_shape = kernel.shape
    kernel = kernel.reshape(src_shape)
    src_image = kernel.to(torch.complex64)
    target_image = heatmaps.to(torch.complex64)
    src_freq = torch.fft.fft2(scr_image)
    target_freq = torch.fft.fft2(target_image)

    shape = src_freq.shape[1:3]
    shape = shape.reshape(shape = (1,2))
    shape = torch.to(torch.float32)
    shape = torch.tile(input = shape, shape = (target_freq.shape[0], 1))
    image_product = src_freq * torch.conj(target_freq)
    cross_correlation = torch.fft.ifft2(image_product)

    maxima = find_maxima(torch.abs(cross_correlation))
    midpoints = fix(shape.to(torch.float32) / 2)

    shifts = maxima
    shifts = torch.where(shifts > midpoints, shifts - shape, shifts)
    shifts = torch.round(shifts * upsample_factor) / upsample_factor

    upsampled_region_size = torch.ceil(upsample_factor * 1.5)
    dftshift = fix(upsampled_region_size / 2.0)
    normalization = src_freq[0].size.to(torch.float32)
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

    return shifts
