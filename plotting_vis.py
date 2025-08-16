import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, DraggableOffsetBox
from matplotlib.backend_bases import MouseButton
import numpy as np
from scipy.spatial import KDTree
from IPython.display import display
import torch

def plot_with_annotations(features, labels: list[str], images: list[torch.Tensor], title=None, figsize=(12,10), cmap=None):
    assert features.shape[0] == len(images), 'need an image for all features'
    assert features.shape[0] == len(labels), 'need a label for all features'

    tree = KDTree(features)
    unique_labels = np.unique(labels)

    mouseover_radius = 0.4
    padding_percent = 0.3
    img_zoom = 0.63

    plt.close('all')
    fig = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    ax = plt.gca()

    for label in reversed(unique_labels):
        i = np.where(np.array(labels) == label)
        plt.scatter(
            features[i, 0], features[i, 1], label=label, cmap=cmap
        )
    pp = padding_percent/2.0
    plt.subplots_adjust(left=pp, right=1-pp, top=1-pp, bottom=pp)

    # handles, lg_labels = ax.get_legend_handles_labels()
    # sorted_handles_labels = sorted(zip(lg_labels, handles), key=lambda x: x[0])
    # lg_labels, handles = zip(*sorted_handles_labels)
    # legend = plt.legend(handles, lg_labels, loc='lower left', fontsize='20', markerscale=2.0)

    legend = plt.legend()
    legend_colors = [handle.get_fc() for handle in legend.legend_handles]

    imagebox = OffsetImage(np.zeros((1,1,3)), zoom=img_zoom)
    imagebox.image.axes = ax

    xybox = (50., 50.)
    ab = AnnotationBbox(imagebox, (0,0), xybox=xybox, xycoords='data', boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="simple"))
    ax.add_artist(ab)
    ab.set_visible(False)

    pinned_indices = [False] * len(labels)
    pinned_imageboxes = []
    pinned_abs = []

    target = None
    lmb_down = False
    dragging_box = -1
    hovering_box = False


    def hover(event):
        nonlocal target, hovering_box, dragging_box

        if dragging_box >= 0:
            (fig_x, fig_y) = (event.x, event.y)
            xy = pinned_abs[dragging_box].xy
            pinned_abs[dragging_box].boxcoords = "figure pixels"
            pinned_abs[dragging_box].xybox = (fig_x - xy[0], fig_y - xy[1])
            fig.canvas.draw_idle()
            return

        if target is not None:
            target.remove()
            target = None

        (x,y) = (event.xdata, event.ydata)
        if x is None or y is None:
            return

        for annotation in pinned_imageboxes:
            if annotation is None:
                continue
            bbox = annotation.get_window_extent(fig.canvas.get_renderer())
            if bbox.contains(event.x, event.y):
                ab.set_visible(False)
                hovering_box = True
                fig.canvas.set_cursor(4)
                fig.canvas.draw_idle()
                return
        
        if hovering_box:
            fig.canvas.set_cursor(1)
            hovering_box = False
        
        (d,i) = tree.query([x,y])

        if d < mouseover_radius and pinned_indices[i] == False:
            (fx, fy) = (features[i,0], features[i,1])
            ab.set_visible(True)
            ab.xy = (fx,fy)
            imagebox.set_data(images[i].numpy().transpose(1,2,0))
        else:
            ab.set_visible(False)

        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal lmb_down, pinned_abs, pinned_abs, pinned_indices, dragging_box
        if event.button is not MouseButton.LEFT:
            return
        
        lmb_down = True

        for (i, annotation) in enumerate(pinned_abs):
            if annotation is None:
                continue
            bbox = annotation.get_window_extent(fig.canvas.get_renderer())
            if bbox.contains(event.x, event.y):
                dragging_box = i
                return
        
        if ax.contains(event)[0]:
            (x,y) = (event.xdata, event.ydata)
            if x is None or y is None:
                return
            
            (d,i) = tree.query([x,y])
            if (d < mouseover_radius) and (pinned_indices[i] == False):
                (fx, fy) = (features[i,0], features[i,1])
                pinned_indices[i] = True

                new_imagebox = OffsetImage(images[i].numpy().transpose(1,2,0), zoom=img_zoom) # change size of pinned images
                new_imagebox.image.axes = ax

                display(f'selected image {i}')
                
                new_ab = AnnotationBbox(new_imagebox, xy=(fx,fy), xybox=xybox, xycoords='data', boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="simple", facecolor="black"))
                ax.add_artist(new_ab)
                
                pinned_imageboxes.append(new_imagebox)
                pinned_abs.append(new_ab)
            elif (d < mouseover_radius) and (pinned_indices[i] == True):
                (fx, fy) = (features[i,0], features[i,1])
                for (idx, abox) in enumerate(pinned_abs):
                    if abox is None:
                        continue
                    if abox.xy == (fx, fy):
                        pinned_abs[idx].remove()
                        pinned_abs[idx] = None
                        pinned_imageboxes[idx] = None
                        pinned_indices[idx] = False



    def on_release(event):
        nonlocal lmb_down, dragging_box
        if event.button is not MouseButton.LEFT:
            return
        
        lmb_down = False
        
        if dragging_box >= 0:
            dragging_box = -1
        
        if ax.contains(event)[0]:
            return

    plt.connect('motion_notify_event', hover)
    plt.connect('button_press_event', on_click)
    plt.connect('button_release_event', on_release)

    plt.show()


def main():
    import torch
    n = 20

    features = np.random.normal(0, 1.0, size=(n, 2))
    labels = ['test'] * n
    images = [torch.rand(3,32,32) for _ in range(n)]

    plot_with_annotations(features, labels, images)

if __name__ == "__main__":
    main()