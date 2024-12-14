"""
Reconstruction module

methods for handling playback
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import ipywidgets as wd

def interactive(recon_data,palette='jet'):
    """
    Using matplotlib widget, slider control of Z-height
    """
    fig,axes = plt.subplots()

    plt.imshow(recon_data[0],palette)
    plt.colorbar()

    slider=wd.IntSlider(
        value=0, #intial image index
        min=0,
        max=recon_data.shape[0]-1, #video shouldn't play more than time limit
    )
    play_button=wd.Play(
        value=0, #intial image index
        min=0,
        max=recon_data.shape[0]-1, #video shouldn't play more than time limit
        step=1,
        interval=200, #referesh interval in ms
        description="Press play",
    )
    wd.jslink((play_button,"value"),(slider,"value"))

    def slider_update(change):
        axes.imshow(
            recon_data[change.new], 
            cmap=palette, 
            origin='lower', 
            vmax=recon_data.max(), 
            vmin=recon_data.min()
        )
        fig.canvas.draw_idle()
        plt.suptitle(f'Time: {slider.value}')

    slider.observe(slider_update, "value")
    out=wd.Output()
    app=wd.VBox([wd.HBox([play_button,slider]),out])
    display(app)
    
def gif(recon_data,interval, savefile):
    """
    Saves .gif of reconstruction, going through z-heights
    Can be rendered in jupyterlab using from IPython.display import HTML
    
    Args:
        recon_data: with shape (z,x,y)
        interval: frame rate for gif, in ms
        
    Returns:
        html5 <video> tag.
    """
    fig, ax = plt.subplots()

    im = ax.imshow(recon_data[0,:,:])

    def init():
        im.set_data(recon_data[0,:,:])
        return (im,)

    # animation function. This is called sequentially
    def animate(i):
        data_slice = recon_data[i,:,:]
        im.set_data(data_slice)
        im.set(clim=[recon_data.min(),recon_data.max()],cmap='inferno')
        ax.set_title(f'Z Height: {i}')
        return (im,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=recon.shape[0], interval=interval, blit=True)
    anim.save(savefile,dpi=300)
    return anim.to_html5_video()