import numpy as np
from scipy import ndimage
import imageio
import matplotlib.pyplot as plt


def generate_wood_face(seed, resolution=(400, 400),
                       depth=1000,
                       early_wood_width_range=(3, 6),
                       early_wood_gray_range=(175, 200),
                       late_wood_width_range=(3, 6),
                       late_wood_gray_range=(50, 75),
                       rot_deg=0):
    if rot_deg != 0:
        cut_resolution = resolution
        cut_depth = depth
        scaling_factor = np.abs(np.cos(rot_deg)) + np.abs(np.sin(rot_deg))
        resolution = tuple(int(x*scaling_factor) for x in cut_resolution)
        depth = scaling_factor * cut_depth

    image = np.zeros(resolution, dtype=np.uint8)
    np.random.seed(seed)

    width, height = resolution
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    ring_size = int(np.sqrt((width + height) ** 2))
    use_early_wood = np.random.choice([True, False])

    while ring_size > 0:
        if use_early_wood:
            min_ring_width, max_ring_width = early_wood_width_range
            min_ring_gray, max_ring_grey = early_wood_gray_range
        else:
            min_ring_width, max_ring_width = late_wood_width_range
            min_ring_gray, max_ring_grey = late_wood_gray_range
        ring_width = np.random.randint(min_ring_width, max_ring_width+1)
        ring_color = np.random.randint(min_ring_gray, max_ring_grey+1)

        x_min = max(0, cx - ring_size)
        x_max = min(width, cx + ring_size)
        y_min = max(0, cy - ring_size)
        y_max = min(height, cy + ring_size)
        y_indices, x_indices = np.ogrid[y_min:y_max, x_min:x_max]
        distances_sq = (x_indices - cx)**2 + (y_indices - cy)**2
        mask = distances_sq <= ring_size**2
        image[y_min:y_max, x_min:x_max][mask] = ring_color

        ring_size -= ring_width
        use_early_wood = ~use_early_wood

    image3d = np.repeat(image[None, ...], depth, axis=0)
    if rot_deg != 0:
        rot_image3d = ndimage.rotate(image3d, angle=rot_deg, reshape=False, axes=(1, 0))
        cut_width, cut_height = cut_resolution
        cut_rot_image3d = rot_image3d[int(depth/2-cut_depth/2):int(depth/2+cut_depth/2),
                                      int(width/2-cut_width/2):int(width/2+cut_width/2),
                                      int(height/2-cut_height/2):int(height/2+cut_height/2)]
        return cut_rot_image3d
    else:
        return image3d


# Example usage:
if __name__ == "__main__":
    image = generate_wood_face(seed=0, rot_deg=20)
    # plt.imshow(image[-1], cmap="gray")
    # plt.imshow(rot_image3d[-1], cmap="gray")
    # plt.show()
    imageio.imwrite('rings0.png', image[0])
    imageio.imwrite('rings200.png', image[200])
    imageio.imwrite('rings400.png', image[400])
    imageio.imwrite('rings600.png', image[600])
    imageio.imwrite('rings800.png', image[800])
    imageio.imwrite('rings999.png', image[999])
