import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from skimage import transform, util


def generate_wood_face(seed, resolution=(400, 300),
                       depth=1000,
                       early_wood_width_range=(3, 6),
                       early_wood_gray_range=(175, 200),
                       late_wood_width_range=(3, 6),
                       late_wood_gray_range=(50, 75),
                       rot_deg=0):
    width, height = resolution
    image = np.zeros((height, width), dtype=np.uint8)
    np.random.seed(seed)

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

    if rot_deg != 0:
        rot_rad = rot_deg * np.pi/180
        b1 = np.tan(rot_rad) * depth
        b = height + b1
        d = np.cos(rot_rad) * b

        resize_height = b
        resize_width = d / height * width

        image = util.img_as_ubyte(transform.resize(image, (resize_height, resize_width)))
        image3d = np.array([image[int(i*(resize_height-height)/depth):int(height+i *
                           (resize_height-height)/depth), int(resize_width/2-width/2):int(resize_width/2+width/2)] for i in range(depth)])
    else:
        image3d = np.repeat(image[None, ...], depth, axis=0)
    return image3d


def interactive_slice_viewer(image3d):
    """
    Displays an interactive slider to scroll through slices of a 3D greyscale array along different axes.

    Parameters:
    - image3d: 3D NumPy array (greyscale values).
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3, left=0.2)

    # Initial view settings
    current_axis = 0  # 0 = axial, 1 = coronal, 2 = sagittal
    slice_index = 0
    max_slices = [image3d.shape[0] - 1, image3d.shape[1] - 1, image3d.shape[2] - 1]
    dimensions = [(image3d.shape[1], image3d.shape[2]), (image3d.shape[0],
                                                         image3d.shape[2]), (image3d.shape[0], image3d.shape[1])]

    slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])

    radio_ax = plt.axes([0.01, 0.3, 0.15, 0.2])
    radio_buttons = RadioButtons(radio_ax, ('Axial', 'Coronal', 'Sagittal'))

    slice_slider, img_display = None, None

    def update_slice(val):
        slice_idx = int(slice_slider.val)
        if current_axis == 0:
            img_display.set_data(image3d[slice_idx, :, :])
        elif current_axis == 1:
            img_display.set_data(image3d[:, slice_idx, :])
        elif current_axis == 2:
            img_display.set_data(image3d[:, :, slice_idx])
        fig.canvas.draw_idle()

    def update_axis(label):
        nonlocal current_axis
        nonlocal img_display
        nonlocal slice_slider

        if label == 'Axial':
            current_axis = 0
        elif label == 'Coronal':
            current_axis = 1
        elif label == 'Sagittal':
            current_axis = 2

        slider_ax.clear()
        slice_slider = Slider(
            ax=slider_ax,
            label='Slice Index',
            valmin=0,
            valmax=max_slices[current_axis],
            valinit=slice_index,
            valstep=1
        )
        slice_slider.on_changed(update_slice)

        img_display = ax.imshow(image3d[0, :, :] if current_axis == 0 else image3d[:, 0, :]
                                if current_axis == 1 else image3d[:, :, 0], cmap='gray')
        ax.set_xlim([0, dimensions[current_axis][1]])
        ax.set_ylim([dimensions[current_axis][0], 0])
        fig.canvas.draw_idle()

    radio_buttons.on_clicked(update_axis)
    update_axis('Axial')
    plt.show()


# Example usage:
if __name__ == "__main__":
    image3d = generate_wood_face(seed=0, rot_deg=5, depth=1000)
    interactive_slice_viewer(image3d)
