import PIL
import sklearn
from utils.glviewer import glumpy_viewer, glumpy
def show():
    Y = [m['label'] for m in self.meta]
    glumpy_viewer(
            img_array=CIFAR10._pixels,
            arrays_to_print=[Y],
            cmap=glumpy.colormap.Grey,
            window_shape=(32 * 2, 32 * 2))
