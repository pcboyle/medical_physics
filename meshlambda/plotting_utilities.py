import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import axes_size
import colorcet as cm
import os
import time
import subprocess
import glob
import utilities as util

####################################################
#
# Set plot defaults
#
####################################################

SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', family='monospace')
plt.rc('font', weight='light')
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Config located at 'C:\\Users\\peter\\Anaconda3\\lib\\site-packages\\matplotlib\\mpl-data\\matplotlibrc'

####################################################
#
# Visualization Functions.
#
####################################################

def force_aspect(axis, aspect = 1):
    """
    aspect: float or int
        Aspect ratio of plot in vertical/horizontal. 
    """
    im = axis.get_images()
    extent =  im[0].get_extent()
    
    vertical = abs((extent[3]-extent[2]))
    horizontal = abs((extent[1]-extent[0]))
    
    ratio = (horizontal/vertical) * aspect
    
    axis.set_aspect(ratio)
    
    return ratio

def add_colorbar(axis, im, fig, aspect = 1, cbar_label = '', loc = 'right', orientation = 'vertical', 
                 tick_param_kwargs = {}, xlabel_kwargs = {}, ylabel_kwargs = {}):
    """
    
    Function to add colorbar to a given axis. 
    
    Parameters:
    ----------
    axis: matplotlib axis
        Axis to add colorbar to. 
    
    im: matplotlib image
        Image to use to determine colorbar values and cmap. 
    
    fig: matplotlib figure
        Figure to add colorbar to. 
    
    cbar_label: string
        Label for the colorbar. 
        
    loc: string
        Location of the colorbar. 
        Default is 'right'. 
        
    orientation: string
        Orientation of the colorbar. If loc is 'top' or 'bottom', reorients to be horizontal. 
        Default is 'vertical'.
    
    tick_param_kwargs: dictionary
        Contains all arguments passed to tick_params function. 
        Example includes: 
            {'axis':'x', 'bottom':False, 'top':True, 'labelbottom':False, 'labeltop':True}
    
    xlabel_kwargs: dictionary
        Contains all arguments passed to set_xlabel function. 
        Example includes:
             {'fontsize':15, 'x':0.5, 'labelpad':-55.0}\
             
    ylabel_kwargs: dictionary
        Same as xlabel_kwargs. 
        
    Outputs:
    -------
    cbar: Axis colorbar
        Can be further manipulated outside of function. 
    
    """
    
    divider = make_axes_locatable(axis)
    size = axes_size.AxesY(axis, aspect=aspect/axis.get_data_ratio())
    
    if loc == 'top' or loc == 'bottom':
        orientation = 'horizontal'
        size = axes_size.AxesX(axis, aspect=aspect*axis.get_data_ratio())
        
    pad = axes_size.Fraction(0.05, size)
    
    cax = divider.append_axes(loc, size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation = orientation)
    cax.tick_params(**tick_param_kwargs)
    
    if len(xlabel_kwargs.keys()) > 0 and cbar_label != '':
        cax.set_xlabel(cbar_label, **xlabel_kwargs)
        
    if len(ylabel_kwargs.keys()) > 0 and cbar_label != '':
        cax.set_ylabel(cbar_label, **ylabel_kwargs)
        
    return cbar

def create_animation(file_name_pattern, framerate = 10, search_dir = os.getcwd(), outdir = os.getcwd(), video_name = 'output', video_format = 'mp4', overwrite = True):
    """
    Function that renders a video from a set of images based on a file name pattern. 
    
    Uses ffmpeg and glob. 
    
    Parameters:
    ----------
    file_name_pattern: string
        Glob search pattern used to find all files for video creation. 
        File names should be of the form:
            '<prefix>_%3d.png' 
            where prefix is the plot prefixes and %3d is a 3 digit number corresponding 
            to the frame number. 
    
    framerate: int
        Frames per second of the final video. 
        Default is 10. 
        
    search_dir: string
        Directory to search for images in. 
        Default is current working directory. 
        
    outdir: string
        Directory to save the video in. 
        Default is current working directory. 
        
    video_name: string
        Name of the video. 
        Default is 'output'.
        
    video_format: string
        Format of the video. Options are any that ffmpeg can render. 
        Default is mp4. 
        
    overwrite: Boolean
        If True, will overwrite a video with the same filename and extension specified. 
        If False, will prompt the user for a new filename. 
        Default is True. 
        
    Outputs:
    -------
    None
    
    Saves:
    -----
    video: mp4 file
        Rendered video using the specified settings. 
    
    """
    
    video_filename = '{0}.{1}'.format(video_name, video_format)
    video_path = os.path.join(outdir, video_filename)
    
    if os.path.exists(video_path) & overwrite:
        print('Overwriting {0}!'.format(video_filename))
        os.remove(video_path)
        
    elif os.path.exists(video_path) & (not overwrite):
        print('Video with name {0} exists! Please enter new video name!'.format(video_filename))
        new_video_name = input()
        video_filename = '{0}.{1}'.format(new_video_name, video_format)
        video_path = os.path.join(outdir, video_filename)
        
    else:
        print('Saving {0}.'.format(video_filename))
        
    search_pattern = os.path.join(search_dir, file_name_pattern)
    
    command_list = ['ffmpeg', '-framerate', framerate, '-i', search_pattern, video_path]
    command = ['{0}'.format(comm) for comm in command_list]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    
    process.wait()
    
    print('Video saved at {0}.'.format(video_path))
    
    return

def plot_image_array(image_array, dim_axis = 2, show_plot = True, save_plot = False, 
                     plot_name = 'plot', plot_directory = os.getcwd(), plot_dpi = 100, 
                     plot_min = 0, plot_max = 2000):
    
    """
    Function to plot a 3D image as a grid of 2D plots. Each plot is formatted
    to be as square as possible, with empty subplot axes removed.
    
    If the shape of the 3D image is (50, 50, 100) and we plot along the 3rd axis 
    (ie, z axis) with length 100, the final plot will be a 10 by 10 grid with 
    z = 0 at the top left and z = 99 at the bottom right. 

    Can save a plot with filename <plot_name>_<horizontal_axis>_<vertical_axis>.png.
    If dim_axis = 2, then the plot filename will be <plot_name>_x_y.png.
    
    Parameters:
    ----------
    image_array: numpy array
        Numpy image array. Must be 3 dimensional, preferably as (x, y, z). 
        
    dim_axis: int
        Axis to plot sequence along. If dim_axis = 2, will plot the series of x,y planes along z. 
        Default is 2. 
        
    show_plot: Boolean
        If True, show the resulting plot. 
        Default is True.
        
    save_plot: Boolen
        If True, will save the plot under the appropriate plot name and plot directory. 
        Default is False. 
        
    plot_name: string
        Prefix for the plot filename. 
        Default is 'plot'.
        
    plot_directory: string
        Directory to save the plot in. 
        Default is current working directory. 
        
    plot_dpi: int
        Resolution of the plot in pixels per inch. 
        Default is 100. 
    
    Outputs:
    -------
    None
    
    Saves:
    -----
    Subplot grid, if desired.     
    """
    
    # [x_dim, y_dim, z_dim]
    dimensions = np.shape(image_array)
    
    # Swap axes of array depending on chosen plot axis in order to ensure
    # that plot is along 3rd axis. 
    if dim_axis == 0: 
        image_array = np.swapaxes(image_array, 0, 2) # x -> z = (z, y, x)
        image_array = np.swapaxes(image_array, 0, 1) # z -> y = (y, z, x)
        axes_names = ['y', 'z']
    
    if dim_axis == 1:
        image_array = np.swapaxes(image_array, 1, 2) # y -> z = (x, z, y)
        axes_names = ['x', 'z']
        
    if dim_axis == 2: # This if statement here for clarity and completion. 
        axes_names = ['x', 'y']
    
    # Get subplots in each direction, make plot as square as possible. 
    n_subplots_rows = int(np.floor(np.sqrt(dimensions[dim_axis])))
    n_subplots_cols = int(np.ceil(np.sqrt(dimensions[dim_axis])))
    
    plt.close('all')

    fig = plt.figure(figsize = (n_subplots_cols*2, n_subplots_rows*2), dpi = plot_dpi)
    plt.style.use('dark_background')
    fig.patch.set_facecolor('k')
    
    grid = AxesGrid(fig, 111,
                    nrows_ncols = (n_subplots_rows, n_subplots_cols),
                    axes_pad = 0.05,
                    cbar_mode = 'single',
                    cbar_location = 'right',
                    cbar_pad = 0.1,
                    cbar_size = '2%')
    
    vmin = plot_min
    vmax = plot_max
    
    for ax_number, ax in enumerate(grid):
        
        if ax_number < dimensions[dim_axis]:
            im = ax.imshow(image_array[:,:,ax_number], cmap = cm.m_fire, 
                           vmin = vmin, vmax = vmax, origin = 'lower')
            ax.set_aspect('equal')
            ax.axis('off')
        else:
            ax.axis('off')
            
    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    
    plot_title = '{0}_{1}_{2}'.format(plot_name, *axes_names)
    plt.suptitle(plot_title, fontsize = 25, y = 0.905)
    
    if save_plot:
        plot_filename = '{0}.png'.format(plot_title)
        plot_pathname = os.path.join(plot_directory, plot_filename)
        plt.savefig(plot_pathname, dpi = 'figure', facecolor = 'k', edgecolor = 'k', bbox_inches = 'tight')
    
    if show_plot:
        plt.show()
    
    else:
        plt.close('all')
        
    return

def plot_image_sequence(image_array, dim_axis = 2, show_plot = False, save_plot = True, 
                        prefix = 'plot', plot_directory = os.getcwd()):
    
    """
    Function to plot the time sequence of a 3D image array. Uses plot_image_array function. 
    Assumes axes are like [x, y, z, time]. 
    
    Saves plots as <prefix>_<time_point>_<horizontal_axis>_<vertical_axis>.png, where
    time_point is the integer count of the images, and horizontal and vertical axes can 
    be x, y, or z depending on the selected dim_axis. If dim_axis = 2, then the plot 
    filenames will be <prefix>_<time_point>_x_y.png.
    
    Parameters:
    ----------
    image_array: Numpy array
        4D array to be plotted. 
        
    dim_axis: int
        Axis to plot along. Must be 0, 1, or 2. 
        Default is 2. 
        
    show_plot: Boolean
        If True, will print the plot from each timepoint to screen. 
        Default is False.
        
    save_plot: Boolean
        If True, will save each plot with the given prefix and count.
        Default is True. 
        
    prefix: string
        Prefix for the saved filenames. 
        Default is 'plot'.
        
    plot_directory: string
        Directory to save the plots in. 
        Default is current working directory. 
        
    Outputs:
    -------
    None
    
    Saves:
    -----
    Sequence of subplot grids for each timepoint. 
    """
    
    plot_dir = outdir_check(plot_directory)
    
    time_points = np.shape(image_array)[3]
    
    leading_zeros = len(str(time_points))
    plot_min = np.min(image_array)
    plot_max = np.max(image_array)
    
    for t_point in range(time_points):
        
        print('Plotting image {0} of {1}.'.format(t_point + 1, time_points), end = '\r')
        
        plot_name = '{0}_{1}'.format(prefix, str(t_point).zfill(leading_zeros))
        plot_image_array(image_array[:,:,:,t_point], dim_axis = dim_axis, show_plot = show_plot, 
                         save_plot = save_plot, plot_name = plot_name, plot_directory = plot_directory, 
                         plot_min = plot_min, plot_max = plot_max)
    
    return

def save_sequence_as_video(png_directory, prefix = 'sequence', dim_axis = 2, framerate = 4):
    """
    Function to save the image sequences produced by plot_image_sequence as a video. 
    
    Uses ffmpeg, glob, and create_animation.
    
    Parameters:
    ----------
    png_directory: string
        Directory where the image sequences are saved. 
        
    prefix: string
        Prefix for the video filename. 
        Default is 'sequence'. 
        
    dim_axis: int
        Axis that was plotted along. Must be 0, 1, or 2. 
        Default is 2. 
        
    framerate: int
        Frames per second for the video. 
        Default is 4.
        
    Outputs:
    -------
    None
    
    Saves: 
    -----
    video: mp4 file
        The video will be saved as '<prefix>_<suffix>.mp4', where
        prefix is the same as above and suffix can be 'y_z', 'x_z', 
        or 'x_y' for dim_axis = 0, 1, or 2, respectively.
    """
    
    suffixes = ['y_z', 'x_z', 'x_y']
    
    glob_pattern = '*{0}.png'.format(suffixes[dim_axis])

    video_filename = '{0}_{1}'.format(prefix, suffixes[dim_axis])
    
    create_animation(glob_pattern, framerate = framerate, search_dir = png_directory, 
                     outdir = png_directory, video_name = video_filename, video_format = 'mp4', 
                     overwrite = True)
    
    return

def visualize_dict_sequences(data_dict, dict_keys = None, framerates = (1, 2, 4), plot_axes = (0, 1, 2), 
                             plot_directory = os.getcwd()):
    """
    Function to plot entirety of a given data dictionary and create a video for each sequence. Tests whether given
    dictionary entry has dimensions necessary. 
    
    Filenames taken from the data_dict keys. Creates a parent plot directory and stores each sequence in a folder
    determined by the data_dict keys. 
    
    Uses ffmpeg to create video. 
    
    Parameters:
    ----------
    data_dict: dictionary
        Data dictionary containing appropriate keys and data to plot. 
        
    dict_keys: string
        If not None, will use the specified keys instead of all keys. 
        Default is None. 
        
    framerates: tuple
        Saved video framerates in frames per second. 
        Default is (1, 2, 4).
        
    plot_axes: tuple of ints
        Axes to plot along. Can be (1), (1, 2), (1, 3), etc. 
        If plotting axis 0, eg x axis, images will be of y-z plane. 
        Default is (0, 1, 2). 
        
    plot_directory: string
        Parent location for the saved plots and videos. 
        Default is current working directory. 
        
    Outputs:
    -------
    Plots and video saved at in the plot_directory. 
    """
    
    if dict_keys:
        data_dict_keys = [dict_keys]
        
    else:
        data_dict_keys = list(data_dict.keys())
    
    outdir_check(plot_directory)
    
    for data_key in data_dict_keys:
        image_data = data_dict[data_key]
        
        # Check dimension of image data. 
        # Skip over entries with a single entry or with dimension < 3. 
        # Not sure if all these if statements are necessary. 
        if np.size(image_data) == 1:
            print('*'*25)
            print('Key {0} excluded for being a single entry.'.format(data_key))
            print('*'*25)
            print('')
            continue
        
        if len(np.shape(image_data)) > 2:
            print('*'*25)
            print('Working on {0}.'.format(data_key))
            print('*'*25)
            print('')
            
        else:
            print('*'*25)
            print('Key {0} excluded for having dimensions < 3.'.format(data_key))
            print('*'*25)
            print('')
            continue
            
        prefix = '{0}'.format(data_key)
        
        for dim_axis in plot_axes:
            print('-'*25)
            print('Working on axis {0}'.format(dim_axis))
            print('-'*25)
            
            plot_image_sequence(image_data, dim_axis = dim_axis, show_plot = False,
                                save_plot = True, prefix = data_key, plot_directory = plot_directory)
            
            for framerate in framerates:
                save_sequence_as_video(png_directory, prefix = prefix, dim_axis = dim_axis, framerate = framerate)
                
    return

def plot_shelf_check_plot(data_dict, data_key, plot_rows = 1, original_reference = 0, smooth_kernal = 0.5):
    """
    
    """
    plt.close('all')
    
    if plot_rows == 1:
        
        fig, ax = plt.subplots(1, 3, figsize = (16, 7))
        
        ax[0].imshow(data_dict['data']['reference'][10,:,:], cmap = cm.m_fire, origin = 'lower', vmin = np.min(original_reference), vmax = np.max(original_reference))
        ax[1].imshow(data_dict['data']['comparison'][10,:,:], cmap = cm.m_fire, origin = 'lower', vmin = np.min(original_reference), vmax = np.max(original_reference))
        ax[1].set_title('Clean, Noise Level = {0}'.format(data_key), fontsize = 15)
        
        x_range = np.linspace(0, len(original_reference), len(original_reference))
        
        for ref_x in data_dict['data']['comparison'][0]:
            ax[2].plot(x_range, ref_x)
    
    else:
        
        fig, ax = plt.subplots(2, 3, figsize = (16, 14))
        
        ax[0][0].imshow(data_dict['data']['reference'][10,:,:], cmap = cm.m_fire, origin = 'lower', vmin = np.min(original_reference), vmax = np.max(original_reference))
        ax[0][1].imshow(data_dict['data']['comparison'][10,:,:], cmap = cm.m_fire, origin = 'lower', vmin = np.min(original_reference), vmax = np.max(original_reference))
        ax[0][1].set_title('Unsmoothed, Noise Level = {0}'.format(data_key), fontsize = 15)
        
        x_range = np.linspace(0, len(original_reference), len(original_reference))
        
        for ref_x in data_dict['data']['comparison'][0]:
            ax[0][2].plot(x_range, ref_x)
            
        ax[1][0].imshow(data_dict['data']['smooth_reference'][10,:,:], cmap = cm.m_fire, origin = 'lower', vmin = np.min(original_reference), vmax = np.max(original_reference))
        ax[1][1].imshow(data_dict['data']['smooth_comparison'][10,:,:], cmap = cm.m_fire, origin = 'lower', vmin = np.min(original_reference), vmax = np.max(original_reference))
        ax[1][1].set_title('Smoothed, Kernal Size = {0}'.format(smooth_kernal), fontsize = 15)
        
        x_range = np.linspace(0, len(original_reference), len(original_reference))
        
        for ref_x in data_dict['data']['smooth_comparison'][0]:
            ax[1][2].plot(x_range, ref_x)
            
    plt.tight_layout()
    
    return

def plot_shelf_sequence(array_3d, cmap = 'viridis', origin = 'upper', vmin = None, vmax = None, cbar_label = r'$\lambda$', 
                        cbar_label_loc = 0.5, x_axis_labels = ['y', 'x', 'x'], y_axis_labels = ['z', 'z', 'y'], 
                        axis_text = ['x', 'y', 'z'], axis_labelsize = 20, axis_textsize = 15,
                        axis_text_loc = [80, 95], axis_textstroke = True, plot_dir = os.getcwd(), prefix = ''):
    
    plot_dir = outdir_check(plot_dir)
    prefix, prefix_folder = prefix_check(prefix)
    
    if type(vmin) == type(None):
        vmin = np.min(array_3d)
    
    if type(vmax) == type(None):
        vmax = np.max(array_3d)
        
    z_fill_len = len('{0}'.format(len(array_3d)))
    
    for arr_ind in range(len(array_3d)):
        print('Plotting image {0} of {1}.'.format(arr_ind + 1, len(array_3d)), end = '\r')
        
        plt.close('all')

        fig, ax = plt.subplots(1, 3, figsize = (18, 7))
        
        for ax_ind in range(3):
            
            if ax_ind == 0:
                image = array_3d[arr_ind, :, :].T
            
            elif ax_ind == 1:
                image = array_3d[:,arr_ind,:].T
                
            else:
                image = array_3d[:,:,arr_ind].T
                
            x_label = x_axis_labels[ax_ind]
            y_label = y_axis_labels[ax_ind]
            
            axis = ax[ax_ind]
            im = axis.imshow(image, cmap = cmap, origin = origin, vmin = vmin, vmax = vmax)
            axis.set_xlabel(x_label, fontsize = axis_labelsize)
            axis.set_ylabel(y_label, fontsize = axis_labelsize)
            
            im_text = axis.text(axis_text_loc[0], axis_text_loc[1], '{0} = {1}'.format(axis_text[ax_ind], arr_ind), c = 'w', fontsize = axis_textsize)
            im_text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
            
            divider = make_axes_locatable(axis)
            cax = divider.append_axes('top', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation = 'horizontal')
            cax.tick_params(axis = 'x', bottom = False, top = True, labelbottom = False, labeltop = True)
            cax.set_xlabel(cbar_label, fontsize = axis_textsize, x = cbar_label_loc, labelpad = -55.0)
            
        plt.tight_layout()
        
        plot_num = '{0}'.format(arr_ind)
        plot_num = plot_num.zfill(z_fill_len)
        plot_name = '{0}{1}.png'.format(prefix, plot_num)
        plot_path = os.path.join(plot_dir, plot_name)
        plt.savefig(plot_path, bbox_inches = 'tight')
    
    plt.close('all')
    
    return

def plot_tube_slice_profile(tube_array, z_slice = 0, fig_dim = 6, wings = True):
    """
    """
    
    plt.close('all')
    
    tube_cross_section_dim = np.shape(tube_array)[0]
    
    tube_center_index = int(np.ceil(tube_cross_section_dim / 2))
    
    tube_slice_values_x = tube_array[tube_center_index, :, z_slice]
    tube_slice_values_y = tube_array[:, tube_center_index, z_slice]
    tube_slice_positions = range(0, tube_cross_section_dim)
    
    if wings:
        fig = plt.figure(figsize = (fig_dim, fig_dim))
        gs = fig.add_gridspec(2, 2,  width_ratios=(fig_dim, fig_dim/4), height_ratios=(fig_dim/4, fig_dim),
                              left=0.0, right=1.0, bottom=0.0, top=1.0,
                              wspace=0.05, hspace=0.05)
        
        ax = fig.add_subplot(gs[1, 0])
        ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_y = fig.add_subplot(gs[1, 1], sharey=ax)
        
        ax.imshow(tube_array[:,:,z_slice], origin = 'lower')
        ax_x.plot(tube_slice_positions, tube_slice_values_x, lw = 2, c = 'firebrick', ls = '-')
        ax_y.plot(tube_slice_values_y, tube_slice_positions, lw = 2, c = 'firebrick', ls = '-')
        
        ax.set_xlabel('X Position [Voxels]', fontsize = 15)
        ax.set_ylabel('Y Position [Voxels]', fontsize = 15)
    
        ax_x.set_ylabel('Value [AU]', fontsize = 15)
        ax_y.set_xlabel('Value [AU]', fontsize = 15)
        
        ax.set_xlim(-0.5, tube_cross_section_dim - 0.5)
        ax.set_ylim(-0.5, tube_cross_section_dim - 0.5)
        ax.tick_params(which = 'both', labelsize = 12)
        ax_x.tick_params(which = 'both', bottom = False, labelbottom = False, top = True, labeltop = True, labelsize = 12)
        ax_y.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = True, labelsize = 12)
        
        tick_list = list(range(0, np.shape(tube_array)[0], 5))
    
        ax.set_xticks(tick_list)
        ax.set_yticks(tick_list)
        
    else:
        fig, ax = plt.subplots(figsize = (fig_dim, fig_dim))
        
        ax.imshow(tube_array[:,:,z_slice], origin = 'lower')
        ax.set_xlabel('X Position [Voxels]', fontsize = 15)
        ax.set_ylabel('Y Position [Voxels]', fontsize = 15)
        ax.set_xlim(-0.5, tube_cross_section_dim - 0.5)
        ax.set_ylim(-0.5, tube_cross_section_dim - 0.5)
        ax.tick_params(which = 'both', labelsize = 12)
        
        tick_list = list(range(0, np.shape(tube_array)[0], 5))
    
        ax.set_xticks(tick_list)
        ax.set_yticks(tick_list)
        plt.tight_layout()        
    
    return

def plot_tube_check_plot(data_dict, data_key, plot_rows = 1, original_reference = [0], smooth_kernal = 0.5):
    """
    Function to save a tube check plot. Used in utilities.auto_lambda function. 
    
    Can plot either a 2 panel arranged in 1 row with 2 columns or a 4 panel with 2 rows and 2 columns. 
    Uses plot_tube_2_panel or plot_tube_4_panel, respectively. 
    
    Does not save plot. Plot can be saved using plt.savefig(<plot_name>).
    
    Parameters:
    ----------
    data_dict: dictionary
        Dictionary containing the reference and comparison information, 
        in practice this is the dictionary for a specific noise level. 
        
    data_key: string
        Key for the selected dictionary. Functions as an axis label. 
        
    plot_rows: int
        Number of rows to plot. 
        Default is 1. 
        
    original_reference: Numpy array
        Original reference array. Used for finding the axis limits. 
        Default is [0].
        
    smooth_kernal: float
        Smoothing kernal size. 
        Default is 0.5. 
        
    Outputs:
    -------
    None
    
    Saves:
    -----
    None
    
    """
    
    if plot_rows == 1:
        reference_tube_array = data_dict['data']['reference']
        comparison_tube_array = data_dict['data']['comparison']
        
        plot_tube_2_panel(reference_tube_array, comparison_tube_array, value_max = np.max(original_reference), 
                          z_slice = 0, fig_dim = 6, left_label = 'Clean reference', right_label = 'Comparison')
    
    else:
        tube_1 = data_dict['data']['reference']
        tube_2 = data_dict['data']['comparison']
        tube_3 = data_dict['data']['smooth_reference']
        tube_4 = data_dict['data']['smooth_comparison']
        
        plot_tube_4_panel(tube_1, tube_2, tube_3, tube_4, z_slice = 0, fig_dim = 6, xy_axis_lims = [np.min(original_reference) - 0.05, np.max(original_reference) + 0.05], 
                          axes_labels = ['Reference\nNoise = {0}\nUnsmooth'.format(data_key), 'Comparison\nNoise = {0}\nUnsmooth'.format(data_key), 
                                         'Reference\nNoise = {0}\nSmooth = {1}'.format(data_key, smooth_kernal), 'Comparison\nNoise = {0}\nSmooth = {1}'.format(data_key, smooth_kernal)])
    return

def plot_tube_2_panel(tube_reference, tube_comparison, value_max = 1, z_slice = 0, subplot_dim = 6, left_label = 'Reference', right_label = 'Comparison'):
    """
    Function to plot a 2 panel tube check plot. 
    
    Parameters:
    ----------
    tube_reference: Numpy array
        Reference tube array. 
        
    tube_comparison: Numpy aarray
        Comparison tube array.
        
    value_max: float or int
        Max value for color map. 
        Default is 1.
        
    z_slice: int
        Slice to plot. 
        Default is 0. 
        
    subplot_dim: float or int
        Dimension of the rows and columns. 
        Default is 6. 
        
    left_label: string
        Label for the left subplot. 
        Default is 'Reference'.
        
    right_label: string
        Label for the right subplot.
        Default is 'Comparison'.
        
    Outputs:
    -------
    None
    
    Saves:
    -----
    None
    """
    
    plt.close('all')
    
    fig = plt.figure(figsize = (subplot_dim * 2, subplot_dim))
    
    tube_cross_section_dim = int(np.shape(tube_reference)[0])
    
    position_index = int(tube_cross_section_dim/2)
    z_index = 0
    
    circle_slice_values_x_ref = tube_reference[position_index, :, z_slice]
    circle_slice_values_y_ref = tube_reference[:, position_index, z_slice]
    
    circle_slice_values_x_comp = tube_comparison[position_index, :, z_slice]
    circle_slice_values_y_comp = tube_comparison[:, position_index, z_slice]
    
    circle_slice_positions = np.array(range(0, np.shape(tube_reference)[0]))
    
    gs = fig.add_gridspec(2, 4,  width_ratios=(subplot_dim/4, subplot_dim, subplot_dim, subplot_dim/4), height_ratios=(subplot_dim/4, subplot_dim),
                          left=0.0, right=1.0, bottom=0.0, top=1.0,
                          wspace=0.0, hspace=0.0)
    
    if value_max > 1.1:
        base_multiple = 5
        max_val_multiple = value_max // base_multiple
    
        new_max = base_multiple*(max_val_multiple + 1)
        value_axis_ticklist = np.linspace(np.min(tube_reference), new_max - base_multiple, 3)
            
    else:
        value_axis_ticklist = np.arange(0.5, value_max + 0.5, 0.5)
    
    # First set of axes. 
    ax = fig.add_subplot(gs[1, 1])
    ax_x = fig.add_subplot(gs[0, 1], sharex = ax)
    ax_y = fig.add_subplot(gs[1, 0], sharey = ax)
    
    ax.imshow(tube_reference[:,:,z_slice], origin = 'lower', cmap = cm.m_fire, vmax = np.ceil(value_max))
    ax_x.plot(circle_slice_positions, circle_slice_values_x_ref, lw = 2, c = 'firebrick', ls = '-')
    ax_y.plot(circle_slice_values_y_ref, circle_slice_positions, lw = 2, c = 'firebrick', ls = '-')
    
    # First set of axes options. value_axis_ticklist = [0.5, 1]
    ax.set_xlabel('X Position [Voxels]', fontsize = 15)
    ax.set_xlim(-0.5, tube_cross_section_dim - 0.5)
    ax.set_ylim(-0.5, tube_cross_section_dim - 0.5)
    ax.tick_params(which = 'both', labelsize = 12, left = False, labelleft = False)
    ax_tick_list = list(range(0, tube_cross_section_dim, 5))
    ax.set_xticks(ax_tick_list)
    ax.set_yticks(ax_tick_list)
    
    ax_x.set_ylabel('Value [AU]', fontsize = 15)
    ax_x.set_yticks(value_axis_ticklist)
    ax_x.tick_params(which = 'both', bottom = False, labelbottom = False, top = True, labeltop = True, labelsize = 12)
    
    ax_y.set_xticks(value_axis_ticklist)
    ax_y.set_xlim(np.flip(ax_y.get_xlim()))
    ax_y.set_xlabel('Value [AU]', fontsize = 15)
    ax_y.set_ylabel('Y Position [Voxels]', fontsize = 15)
    ax_y.tick_params(which = 'both', left = True, labelleft = True, right = False, labelright = False, labelsize = 12)
    
    # Second set of axes
    ax2 = fig.add_subplot(gs[1, 2])
    ax_x2 = fig.add_subplot(gs[0, 2], sharex = ax2)
    ax_y2 = fig.add_subplot(gs[1, 3], sharey = ax2)
    
    ax2.imshow(tube_comparison[:,:,z_slice], origin = 'lower', cmap = cm.m_fire, vmax = np.ceil(value_max))
    ax_x2.plot(circle_slice_positions, circle_slice_values_x_comp, lw = 2, c = 'firebrick', ls = '-')
    ax_y2.plot(circle_slice_values_y_comp, circle_slice_positions, lw = 2, c = 'firebrick', ls = '-')
    
    # Second set of axes options. 
    ax2.set_xlabel('X Position [Voxels]', fontsize = 15)
    ax2.set_xlim(-0.5, tube_cross_section_dim - 0.5)
    ax2.set_ylim(-0.5, tube_cross_section_dim - 0.5)
    ax2.tick_params(which = 'both', labelsize = 12, left = False, labelleft = False)
    ax2_tick_list = list(range(0, tube_cross_section_dim, 5))
    ax2.set_xticks(ax2_tick_list)
    ax2.set_yticks(ax2_tick_list)
    
    ax_x2.tick_params(which = 'both', bottom = False, labelbottom = False, top = True, labeltop = True, labelsize = 12, left = False, labelleft = False)
    
    ax_y2.set_xlabel('Value [AU]', fontsize = 15)
    ax_y2.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = True, labelsize = 12)
    ax_y2.set_xticks(value_axis_ticklist)
    
    ax.text(0.02, 0.95, 'Z slice = {0}'.format(z_index), c = 'w', fontsize = 12, transform = ax.transAxes)
    
    # Title
    fig.suptitle('{0} [left] and {1} [right]'.format(left_label, right_label), fontsize = 20, y = 1.15)   
    
    return 

def plot_tube_4_panel(tube_1, tube_2, tube_3, tube_4, z_slice = 0, subplot_dim = 6, xy_axis_lims = [-0.05, 1.05], wing_type = 'scatter', 
                      axes_labels = ['Unsmooth Reference', 'Unsmooth Comparison', 'Smooth Reference', 'Smooth Comparison'],
                      data_label = 'Value [AU]'):
    """
    
    """

    plt.close('all')

    fig = plt.figure(figsize = (subplot_dim * 2, subplot_dim * 2))
    
    tube_cross_section_dim = int(np.shape(tube_1)[0])
    
    wing_type_list = ['line', 'scatter']
    
    if wing_type not in wing_type_list:
        wing_type = 'line'
        
    position_index = int(tube_cross_section_dim/2)
    
    tube_slice_1_x = tube_1[position_index, :, z_slice]
    tube_slice_1_y = tube_1[:, position_index, z_slice]
    
    tube_slice_2_x = tube_2[position_index, :, z_slice]
    tube_slice_2_y = tube_2[:, position_index, z_slice]
    
    tube_slice_3_x = tube_3[position_index, :, z_slice]
    tube_slice_3_y = tube_3[:, position_index, z_slice]
    
    tube_slice_4_x = tube_4[position_index, :, z_slice]
    tube_slice_4_y = tube_4[:, position_index, z_slice]
        
    if wing_type == 'scatter':
        position_limits = [0, tube_cross_section_dim]
        
        tube_slice_1_x = tube_1[:,:,z_slice]
        tube_slice_1_y = tube_1[:,:,z_slice]
        
        tube_slice_2_x = tube_2[:,:,z_slice]
        tube_slice_2_y = tube_2[:,:,z_slice]
        
        tube_slice_3_x = tube_3[:,:,z_slice]
        tube_slice_3_y = tube_3[:,:,z_slice]
        
        tube_slice_4_x = tube_4[:,:,z_slice]
        tube_slice_4_y = tube_4[:,:,z_slice]
    
    circle_slice_positions = np.array(range(0, tube_cross_section_dim))
    
    gs = fig.add_gridspec(4, 4,  width_ratios=(subplot_dim/4, subplot_dim, subplot_dim, subplot_dim/4), height_ratios=(subplot_dim/4, subplot_dim, subplot_dim, subplot_dim/4),
                          left=0.0, right=1.0, bottom=0.0, top=1.0,
                          wspace=0.0, hspace=0.0)
    
    # First set of axes. 
    ax1 = fig.add_subplot(gs[1, 1])
    ax1_x = fig.add_subplot(gs[0, 1], sharex = ax1)
    ax1_y = fig.add_subplot(gs[1, 0], sharey = ax1)
    ax1_xy = fig.add_subplot(gs[0, 0], sharex = ax1_y, sharey = ax1_x)
    
    # Second set of axes
    ax2 = fig.add_subplot(gs[1, 2])
    ax2_x = fig.add_subplot(gs[0, 2], sharex = ax2)
    ax2_y = fig.add_subplot(gs[1, 3], sharey = ax2)
    ax2_xy = fig.add_subplot(gs[0, 3], sharex = ax2_y, sharey = ax2_x)
    
    # Third set of axes
    ax3 = fig.add_subplot(gs[2, 1])
    ax3_x = fig.add_subplot(gs[3, 1], sharex = ax3)
    ax3_y = fig.add_subplot(gs[2, 0], sharey = ax3)
    ax3_xy = fig.add_subplot(gs[3, 0], sharex = ax3_y, sharey = ax3_x)
    
    # Fourth set of axes
    ax4 = fig.add_subplot(gs[2, 2])
    ax4_x = fig.add_subplot(gs[3, 2], sharex = ax4)
    ax4_y = fig.add_subplot(gs[2, 3], sharey = ax4)
    ax4_xy = fig.add_subplot(gs[3, 3], sharex = ax4_y, sharey = ax4_x)
    
    main_axes = [ax1, ax2, ax3, ax4]
    x_projection_axis = [ax1_x, ax2_x, ax3_x, ax4_x]
    y_projection_axis = [ax1_y, ax2_y, ax3_y, ax4_y]
    xy_projection_axis = [ax1_xy, ax2_xy, ax3_xy, ax4_xy]
    
    main_axis_data = [tube_1[:,:,z_slice], tube_2[:,:,z_slice], tube_3[:,:,z_slice], tube_4[:,:,z_slice]]
    x_projection_axis_data = [[circle_slice_positions, tube_slice_1_x], [circle_slice_positions, tube_slice_2_x],
                              [circle_slice_positions, tube_slice_3_x], [circle_slice_positions, tube_slice_4_x]]
    y_projection_axis_data = [[tube_slice_1_y, circle_slice_positions], [tube_slice_2_y, circle_slice_positions], 
                              [tube_slice_3_y, circle_slice_positions], [tube_slice_4_y, circle_slice_positions]]
    
    main_lims = [-0.5, tube_cross_section_dim - 0.5]
    
    xy_axis_lim_diff = xy_axis_lims[-1] - xy_axis_lims[0]
    
    if xy_axis_lim_diff > 1.5 and xy_axis_lim_diff < 10:
        base_multiple = 0.25
        max_val_multiple = xy_axis_lims[-1] // 0.25

        new_max = base_multiple*(max_val_multiple + 1)
        xy_axis_ticklist = np.round(np.linspace(base_multiple, new_max - base_multiple, 3), decimals = 2)
        
        xy_axis_lims = [-base_multiple, new_max]
        
    elif xy_axis_lims[-1] - xy_axis_lims[0] > 150:
        xy_axis_ticklist = [-90, 0, 90]
        xy_axis_lims = xy_axis_lims
        
    else:
        xy_axis_ticklist = [0.15, .58, 1]
        
    projection_axis_dict = { 1:{'xlim_x': main_lims, 'ylim_x': xy_axis_lims, 'bottom_x': False, 'top_x': True, 'left_x': False, 'right_x': False, 
                            'xlabel_x': ['X Position [Voxels]', 'top'], 'ylabel_x': [data_label, 'left'], 
                            'xlim_y': np.flip(xy_axis_lims), 'ylim_y': main_lims, 'bottom_y': False, 'top_y': False, 'left_y': True, 'right_y': False, 
                            'xlabel_y': [data_label, 'top'], 'ylabel_y': ['Y Position [Voxels]', 'left']}, 
                        
                            2:{'xlim_x': main_lims, 'ylim_x': xy_axis_lims, 'bottom_x': False, 'top_x': True, 'left_x': False, 'right_x': False, 
                            'xlabel_x': ['X Position [Voxels]', 'top'], 'ylabel_x': [data_label, 'right'],
                            'xlim_y': xy_axis_lims,'ylim_y': main_lims, 'bottom_y': False, 'top_y': False, 'left_y': False, 'right_y': True, 
                            'xlabel_y': [data_label, 'top'], 'ylabel_y': ['Y Position [Voxels]', 'right']},
                        
                            3:{'xlim_x': main_lims, 'ylim_x': np.flip(xy_axis_lims), 'bottom_x': True, 'top_x': False, 'left_x': False, 'right_x': False, 
                            'xlabel_x': ['X Position [Voxels]', 'bottom'], 'ylabel_x': [data_label, 'left'], 
                            'xlim_y': np.flip(xy_axis_lims),'ylim_y': main_lims, 'bottom_y': False, 'top_y': False, 'left_y': True, 'right_y': False, 
                            'xlabel_y': [data_label, 'bottom'], 'ylabel_y': ['Y Position [Voxels]', 'left']},
                        
                            4:{'xlim_x': main_lims, 'ylim_x': np.flip(xy_axis_lims), 'bottom_x': True, 'top_x': False, 'left_x': False, 'right_x': False, 
                            'xlabel_x': ['X Position [Voxels]', 'bottom'], 'ylabel_x': [data_label, 'right'], 
                            'xlim_y': xy_axis_lims,'ylim_y': main_lims, 'bottom_y': False, 'top_y': False, 'left_y': False, 'right_y': True, 
                            'xlabel_y': [data_label, 'bottom'], 'ylabel_y': ['Y Position [Voxels]', 'right']} }
    
    for main_ax, main_data in zip(main_axes, main_axis_data):
        main_ax.imshow(main_data, origin = 'lower', cmap = cm.m_fire, vmax = xy_axis_lims[-1])
        main_ax.set_xlim(main_lims)
        main_ax.set_ylim(main_lims)
        main_ax.tick_params(which = 'both', bottom = False, labelbottom = False, left = False, labelleft = False)
        
    for axis_index, (x_axis_data, y_axis_data, x_axis, y_axis, xy_axis) in enumerate(zip(x_projection_axis_data, y_projection_axis_data, x_projection_axis, y_projection_axis, xy_projection_axis)):
        proj_dict = projection_axis_dict[axis_index + 1]
        
        if wing_type == 'line':
            x_axis.plot(*x_axis_data, lw = 2, c = 'firebrick', ls = '-')
            y_axis.plot(*y_axis_data, lw = 2, c = 'firebrick', ls = '-')
        
        elif wing_type == 'scatter':
            x_locs = x_axis_data[0]
            x_vals = x_axis_data[1]
            y_locs = y_axis_data[1]
            y_vals = y_axis_data[0]
            for ax_slice in range(*position_limits):
                x_axis.scatter(x_locs, x_vals[ax_slice, :], c = 'k', alpha = 0.3, marker = '.')
                y_axis.scatter(y_vals[:, ax_slice], y_locs, c = 'k', alpha = 0.3, marker = '.')
            
            x_axis.plot(x_locs, np.mean(x_vals, axis = 0), c = 'royalblue', alpha = 1, ls = '-', marker = '.')
            x_axis.plot(x_locs, np.max(x_vals, axis = 0), c = 'forestgreen', alpha = 1, ls = '--')
            x_axis.plot(x_locs, np.min(x_vals, axis = 0), c = 'firebrick', alpha = 1, ls = '--')
            
            y_axis.plot(np.mean(y_vals, axis = 1), y_locs, c = 'royalblue', alpha = 1, ls = '-', marker = '.')
            y_axis.plot(np.max(y_vals, axis = 1), y_locs, c = 'forestgreen', alpha = 1, ls = '--')
            y_axis.plot(np.min(y_vals, axis = 1), y_locs, c = 'firebrick', alpha = 1, ls = '--')
        
        x_axis.set_yticks(xy_axis_ticklist)
        y_axis.set_xticks(xy_axis_ticklist)
        
        x_axis.set_xlim(proj_dict['xlim_x'])
        x_axis.set_ylim(proj_dict['ylim_x'])
        y_axis.set_xlim(proj_dict['xlim_y'])
        y_axis.set_ylim(proj_dict['ylim_y'])
        
        x_axis.set_xlabel(proj_dict['xlabel_x'][0], fontsize = 15)
        x_axis.set_ylabel('', fontsize = 15)
        y_axis.set_xlabel(proj_dict['xlabel_y'][0], fontsize = 15)
        y_axis.set_ylabel(proj_dict['ylabel_y'][0], fontsize = 15)
        
        x_axis.xaxis.set_label_position(proj_dict['xlabel_x'][1])
        x_axis.yaxis.set_label_position(proj_dict['ylabel_x'][1])
        y_axis.xaxis.set_label_position(proj_dict['xlabel_y'][1])
        y_axis.yaxis.set_label_position(proj_dict['ylabel_y'][1])
        
        x_axis.grid(axis = 'y', color = 'k', linestyle = '-', alpha = 0.2)
        y_axis.grid(axis = 'x', color = 'k', linestyle = '-', alpha = 0.2)
        
        x_axis.tick_params(which = 'both', bottom = proj_dict['bottom_x'], labelbottom = proj_dict['bottom_x'], 
                                           top = proj_dict['top_x'], labeltop = proj_dict['top_x'], 
                                           left = proj_dict['left_x'], labelleft = proj_dict['left_x'],
                                           right = proj_dict['right_x'], labelright = proj_dict['right_x'],
                                           labelsize = 12)
        
        y_axis.tick_params(which = 'both', bottom = proj_dict['bottom_y'], labelbottom = proj_dict['bottom_y'], 
                                           top = proj_dict['top_y'], labeltop = proj_dict['top_y'], 
                                           left = proj_dict['left_y'], labelleft = proj_dict['left_y'],
                                           right = proj_dict['right_y'], labelright = proj_dict['right_y'],
                                           labelsize = 12)
        
        xy_axis.tick_params(which = 'both', bottom = proj_dict['bottom_x'], labelbottom = proj_dict['bottom_x'], 
                                           top = proj_dict['top_x'], labeltop = proj_dict['top_x'], 
                                           left = proj_dict['left_y'], labelleft = proj_dict['left_y'],
                                           right = proj_dict['right_y'], labelright = proj_dict['right_y'],
                                           labelsize = 11, length = 20)
        
        xy_axis.spines['top'].set_visible(proj_dict['bottom_x'])
        xy_axis.spines['bottom'].set_visible(proj_dict['top_x'])
        xy_axis.spines['left'].set_visible(proj_dict['right_y'])
        xy_axis.spines['right'].set_visible(proj_dict['left_y'])
        
        xy_axis.set_xlabel(proj_dict['xlabel_y'][0], fontsize = 15)
        xy_axis.set_ylabel(proj_dict['ylabel_x'][0], fontsize = 15)
        xy_axis.xaxis.set_label_position(proj_dict['xlabel_y'][1])
        xy_axis.yaxis.set_label_position(proj_dict['ylabel_x'][1])
        
        xy_axis.grid(color = 'k', linestyle = '--')
    
    # Text    
    slice_text = ax1.text(0.02, 0.02, 'Z slice = {0}'.format(z_slice), c = 'w', fontsize = 15, transform = ax1.transAxes, verticalalignment = 'bottom')
    slice_text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
    
    for axis, axis_label in zip(main_axes, axes_labels):
        ax_text = axis.text(0.02, 0.98, axis_label, c = 'w', fontsize = 15, transform = axis.transAxes, verticalalignment = 'top')
        ax_text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
    
    return

def setup_4_panel_tube(data_shape = (20, 20), subplot_dim = 6, xy_axis_lims = [-0.05, 1.05], xy_axis_ticklist = [0, 0.5, 1.0], wing_type = 'scatter', 
                       axes_labels = ['Unsmooth Reference', 'Unsmooth Comparison', 'Smooth Reference', 'Smooth Comparison'],
                       data_label = 'Value [AU]', frame_text = 'TEXT', cmap = cm.m_fire):
    """
    Function to set up the 4 panel tube plot. 
    
    Final figure size is 2*subplot_dim by 2*subplot_dim of form:
        [plot 1] [plot 2]
        [plot 3] [plot 4]
    
    Parameters:
    ----------
    data_shape: tuple of ints
        Shape of the data to be plotted. 
        Default is (20, 20).
        
    subplot_dim: float or int
        Dimension of the rows and columns. 
        Default is 6. 
        
    xy_axis_lims: list of float or int
        Limits for the color map and wing plots. 
        Always a list of length 2. 
        Default is [-0.05, 1.05].
        
    xy_axis_ticklist: list of float or int
        Ticks to use for the wing plots. 
        Always a list of length 3. 
        Default is [0, 0.5, 1.0]. 
        
    wing_type: string
        Type of plot for each of the wing plots. 
        Options are 'scatter' and 'line'. 
        Default is 'scatter'.
        
    axes_labels: list of strings
        Label combinations for each of the plots. 
        Always a list of 4 strings, applied in sequence. 
        Default is ['Unsmooth Reference', 'Unsmooth Comparison', 
                    'Smooth Reference', 'Smooth Comparison'].
    
    data_label: string
        Label for the wing plot values. 
        Default is 'Value [AU]'.
        
    frame_text: string
        Placeholder text to apply to each of the 4 subplots. 
        Default is 'TEXT'. 
        
    cmap: colormap
        Colormap to apply to the images. 
        Default is cm.m_fire. 
        
    Outputs:
    -------
    fig: Figure object
        Created figure. 
        
    axis_data_list: list of axis objects
        Created axis objects. 
        
    frame_text_list: list of Text objects
        Created text objects. 
        
    All outputs are used in update_4_panel_tube function. 
    
    Saves:
    -----
    None. 
        
    """
    
    # Set up figure. 
    
    plt.close('all')

    fig = plt.figure(figsize = (subplot_dim * 2, subplot_dim * 2))
    
    gs = fig.add_gridspec(4, 4,  width_ratios=(subplot_dim/4, subplot_dim, subplot_dim, subplot_dim/4), height_ratios=(subplot_dim/4, subplot_dim, subplot_dim, subplot_dim/4),
                          left=0.05, right=0.95, bottom=0.05, top=0.95,
                          wspace=0.0, hspace=0.0)
    
    # First set of axes. 
    ax1 = fig.add_subplot(gs[1, 1])
    ax1_x = fig.add_subplot(gs[0, 1], sharex = ax1)
    ax1_y = fig.add_subplot(gs[1, 0], sharey = ax1)
    ax1_xy = fig.add_subplot(gs[0, 0], sharex = ax1_y, sharey = ax1_x)
    
    # Second set of axes
    ax2 = fig.add_subplot(gs[1, 2])
    ax2_x = fig.add_subplot(gs[0, 2], sharex = ax2)
    ax2_y = fig.add_subplot(gs[1, 3], sharey = ax2)
    ax2_xy = fig.add_subplot(gs[0, 3], sharex = ax2_y, sharey = ax2_x)
    
    # Third set of axes
    ax3 = fig.add_subplot(gs[2, 1])
    ax3_x = fig.add_subplot(gs[3, 1], sharex = ax3)
    ax3_y = fig.add_subplot(gs[2, 0], sharey = ax3)
    ax3_xy = fig.add_subplot(gs[3, 0], sharex = ax3_y, sharey = ax3_x)
    
    # Fourth set of axes
    ax4 = fig.add_subplot(gs[2, 2])
    ax4_x = fig.add_subplot(gs[3, 2], sharex = ax4)
    ax4_y = fig.add_subplot(gs[2, 3], sharey = ax4)
    ax4_xy = fig.add_subplot(gs[3, 3], sharex = ax4_y, sharey = ax4_x)
    
    main_axes = [ax1, ax2, ax3, ax4]
    x_projection_axis = [ax1_x, ax2_x, ax3_x, ax4_x]
    y_projection_axis = [ax1_y, ax2_y, ax3_y, ax4_y]
    xy_projection_axis = [ax1_xy, ax2_xy, ax3_xy, ax4_xy]
    
    fake_x_data = np.random.rand(data_shape[0])
    fake_y_data = np.random.rand(data_shape[1])
    
    circle_slice_positions = np.array(range(0, data_shape[0]))
    position_limits = [0, data_shape[0]]
    
    main_axes_data = []
    x_projection_data = []
    y_projection_data = []
    scatter_mean_data = []
    scatter_min_data = []
    scatter_max_data = []
    
    main_lims = [-0.5, data_shape[0] - 0.5]
    main_tick_list = np.arange(0, data_shape[0] + 5, 5)
        
    projection_axis_dict = { 1:{'xlim_x': main_lims, 'ylim_x': xy_axis_lims, 'bottom_x': False, 'top_x': True, 'left_x': False, 'right_x': False, 
                        'xlabel_x': ['X Position [Voxels]', 'top'], 'ylabel_x': [data_label, 'left'], 
                        'xlim_y': np.flip(xy_axis_lims), 'ylim_y': main_lims, 'bottom_y': False, 'top_y': False, 'left_y': True, 'right_y': False, 
                        'xlabel_y': [data_label, 'top'], 'ylabel_y': ['Y Position [Voxels]', 'left']}, 
                    
                        2:{'xlim_x': main_lims, 'ylim_x': xy_axis_lims, 'bottom_x': False, 'top_x': True, 'left_x': False, 'right_x': False, 
                        'xlabel_x': ['X Position [Voxels]', 'top'], 'ylabel_x': [data_label, 'right'],
                        'xlim_y': xy_axis_lims,'ylim_y': main_lims, 'bottom_y': False, 'top_y': False, 'left_y': False, 'right_y': True, 
                        'xlabel_y': [data_label, 'top'], 'ylabel_y': ['Y Position [Voxels]', 'right']},
                    
                        3:{'xlim_x': main_lims, 'ylim_x': np.flip(xy_axis_lims), 'bottom_x': True, 'top_x': False, 'left_x': False, 'right_x': False, 
                        'xlabel_x': ['X Position [Voxels]', 'bottom'], 'ylabel_x': [data_label, 'left'], 
                        'xlim_y': np.flip(xy_axis_lims),'ylim_y': main_lims, 'bottom_y': False, 'top_y': False, 'left_y': True, 'right_y': False, 
                        'xlabel_y': [data_label, 'bottom'], 'ylabel_y': ['Y Position [Voxels]', 'left']},
                    
                        4:{'xlim_x': main_lims, 'ylim_x': np.flip(xy_axis_lims), 'bottom_x': True, 'top_x': False, 'left_x': False, 'right_x': False, 
                        'xlabel_x': ['X Position [Voxels]', 'bottom'], 'ylabel_x': [data_label, 'right'], 
                        'xlim_y': xy_axis_lims,'ylim_y': main_lims, 'bottom_y': False, 'top_y': False, 'left_y': False, 'right_y': True, 
                        'xlabel_y': [data_label, 'bottom'], 'ylabel_y': ['Y Position [Voxels]', 'right']} }
    
    for main_ax in main_axes:
        main_im = main_ax.imshow(np.zeros(data_shape), origin = 'lower', cmap = cmap, vmin = xy_axis_ticklist[0], vmax = xy_axis_ticklist[-1])
        main_ax.set_xlim(main_lims)
        main_ax.set_ylim(main_lims)
        main_ax.tick_params(which = 'both', bottom = False, labelbottom = False, left = False, labelleft = False)
        
        main_axes_data.append(main_im)
        
    for axis_index, (x_axis, y_axis, xy_axis) in enumerate(zip(x_projection_axis, y_projection_axis, xy_projection_axis)):
        proj_dict = projection_axis_dict[axis_index + 1]
        
        if wing_type == 'line':
            line_x, = x_axis.plot(circle_slice_positions, fake_x_data, lw = 2, c = 'firebrick', ls = '-')
            line_y, = y_axis.plot(fake_y_data, circle_slice_positions, lw = 2, c = 'firebrick', ls = '-')
            
            x_projection_data.append(line_x)
            y_projection_data.append(line_y)
        
        elif wing_type == 'scatter':
            x_locs = circle_slice_positions
            x_vals = fake_x_data
            y_locs = circle_slice_positions
            y_vals = fake_y_data
            
            x_scatter, = x_axis.plot(x_locs, x_vals, c = 'k', alpha = 0.3, marker = '.', lw = 0)
            y_scatter, = y_axis.plot(y_vals, y_locs, c = 'k', alpha = 0.3, marker = '.', lw = 0)
            
            x_mean_lines, = x_axis.plot(x_locs, x_vals, c = 'royalblue', alpha = 1, ls = '-', marker = '.')
            x_max_lines, = x_axis.plot(x_locs, x_vals, c = 'forestgreen', alpha = 1, ls = '--')
            x_min_lines, = x_axis.plot(x_locs, x_vals, c = 'firebrick', alpha = 1, ls = '--')
            
            y_mean_lines, = y_axis.plot(y_vals, y_locs, c = 'royalblue', alpha = 1, ls = '-', marker = '.')
            y_max_lines, = y_axis.plot(y_vals, y_locs, c = 'forestgreen', alpha = 1, ls = '--')
            y_min_lines, = y_axis.plot(y_vals, y_locs, c = 'firebrick', alpha = 1, ls = '--')
            
            xy_mean = [x_mean_lines, y_mean_lines]
            xy_min = [x_min_lines, y_min_lines]
            xy_max = [x_max_lines, y_max_lines]
            
            x_projection_data.append(x_scatter)
            y_projection_data.append(y_scatter)
            scatter_mean_data.append(xy_mean)
            scatter_max_data.append(xy_max)
            scatter_min_data.append(xy_min)
        
        x_axis.set_xticks(main_tick_list)
        x_axis.set_yticks(xy_axis_ticklist)
        y_axis.set_xticks(xy_axis_ticklist)
        y_axis.set_yticks(main_tick_list)
        
        x_axis.set_xlim(proj_dict['xlim_x'])
        x_axis.set_ylim(proj_dict['ylim_x'])
        y_axis.set_xlim(proj_dict['xlim_y'])
        y_axis.set_ylim(proj_dict['ylim_y'])
        
        x_axis.set_xlabel(proj_dict['xlabel_x'][0], fontsize = 15)
        y_axis.set_ylabel(proj_dict['ylabel_y'][0], fontsize = 15)
        
        x_axis.xaxis.set_label_position(proj_dict['xlabel_x'][1])
        x_axis.yaxis.set_label_position(proj_dict['ylabel_x'][1])
        y_axis.xaxis.set_label_position(proj_dict['xlabel_y'][1])
        y_axis.yaxis.set_label_position(proj_dict['ylabel_y'][1])
        
        x_axis.grid(axis = 'y', color = 'k', linestyle = '-', alpha = 0.2)
        y_axis.grid(axis = 'x', color = 'k', linestyle = '-', alpha = 0.2)
        
        x_axis.tick_params(which = 'both', bottom = proj_dict['bottom_x'], labelbottom = proj_dict['bottom_x'], 
                                           top = proj_dict['top_x'], labeltop = proj_dict['top_x'], 
                                           left = proj_dict['left_x'], labelleft = proj_dict['left_x'],
                                           right = proj_dict['right_x'], labelright = proj_dict['right_x'],
                                           labelsize = 12)
        
        y_axis.tick_params(which = 'both', bottom = proj_dict['bottom_y'], labelbottom = proj_dict['bottom_y'], 
                                           top = proj_dict['top_y'], labeltop = proj_dict['top_y'], 
                                           left = proj_dict['left_y'], labelleft = proj_dict['left_y'],
                                           right = proj_dict['right_y'], labelright = proj_dict['right_y'],
                                           labelsize = 12)

        xy_axis.tick_params(which = 'both', bottom = False, labelbottom = False, 
                                            top = False, labeltop = False, 
                                            left = False, labelleft = False,
                                            right = False, labelright = False)
        
        xy_axis.spines['top'].set_visible(proj_dict['bottom_x'])
        xy_axis.spines['bottom'].set_visible(proj_dict['top_x'])
        xy_axis.spines['left'].set_visible(proj_dict['right_y'])
        xy_axis.spines['right'].set_visible(proj_dict['left_y'])
        
        xy_axis.set_xlabel(proj_dict['xlabel_y'][0], fontsize = 15)
        xy_axis.set_ylabel(proj_dict['ylabel_x'][0], fontsize = 15)
        xy_axis.xaxis.set_label_position(proj_dict['xlabel_y'][1])
        xy_axis.yaxis.set_label_position(proj_dict['ylabel_x'][1])
    
    # Set up xy axis ticks. This is a real pain in the ass, but it makes it clean looking. 
    xy_axis_min, xy_axis_max = xy_projection_axis[0].get_ylim()
    xy_axis_tick_locs = [(tick - xy_axis_min)/(xy_axis_max - xy_axis_min) for tick in xy_projection_axis[0].get_yticks()]
    
    ax1_xy_tick_loc_indices = [[2, 0], [1, 1], [0, 2]] # [x, y] index for xy_axis_tick_locs
    ax2_xy_tick_loc_indices = [[0, 0], [1, 1], [2, 2]]
    ax3_xy_tick_loc_indices = [[2, 2], [1, 1], [0, 0]]
    ax4_xy_tick_loc_indices = [[0, 2], [1, 1], [2, 0]]
    
    xy_projection_axis_tick_indices = [ax1_xy_tick_loc_indices, ax2_xy_tick_loc_indices, ax3_xy_tick_loc_indices, ax4_xy_tick_loc_indices]
    
    ax1_xy_tick_adjust = [-0.05, 0.0]
    ax2_xy_tick_adjust = [0.05, 0.0]
    ax3_xy_tick_adjust = [-0.05, 0.0]
    ax4_xy_tick_adjust = [0.05, 0.0]
    
    xy_axis_tick_adjust = [ax1_xy_tick_adjust, ax2_xy_tick_adjust, ax3_xy_tick_adjust, ax4_xy_tick_adjust]
    
    xy1_grid = [[1, 2], [0, 1]]
    xy2_grid = [[0, 1], [0, 1]]
    xy3_grid = [[1, 2], [1, 2]]
    xy4_grid = [[0, 1], [1, 2]]
    
    xy_grid = [xy1_grid, xy2_grid, xy3_grid, xy4_grid]
    
    anchor_list = [['right', 'bottom'], ['left','bottom'], ['right', 'top'], ['left', 'top']]
    
    for xy_axis, xy_tick_indices, xy_tick_adjust, tick_grid, anchor in zip(xy_projection_axis, xy_projection_axis_tick_indices, xy_axis_tick_adjust, xy_grid, anchor_list):
        
        tick_values = xy_axis.get_xticks() # Needed for Grid Lines
        x_tick_limits = [[0, xy_axis_tick_locs[index[0]], 1] for index in xy_tick_indices]
        y_tick_limits = [[0, xy_axis_tick_locs[index[1]], 1] for index in xy_tick_indices]
        
        for tick_label, (tick_ind_1, tick_ind_2), tick, x_limits, y_limits in zip(xy_axis_ticklist, xy_tick_indices, tick_values, x_tick_limits, y_tick_limits):

            x_tick_loc = xy_axis_tick_locs[tick_ind_1]
            y_tick_loc = xy_axis_tick_locs[tick_ind_2]
            
            tick_lab = '{0}'.format(tick_label)
        
            xy_axis.text(x_tick_loc, y_tick_loc, tick_lab, horizontalalignment = anchor[0], verticalalignment = anchor[1],
                         transform=xy_axis.transAxes, bbox = {'boxstyle':'square', 'facecolor':'white', 'alpha':0.0, 'edgecolor':'none', 'pad':0.0})
            
            xmin = x_limits[tick_grid[0][0]]
            xmax = x_limits[tick_grid[0][1]]
            ymin = y_limits[tick_grid[1][0]]
            ymax = y_limits[tick_grid[1][1]]
            
            xy_axis.axhline(tick, xmin = xmin, xmax = xmax, c = 'k', lw = 1, ls = '-', alpha = 0.8)
            xy_axis.axvline(tick, ymin = ymin, ymax = ymax, c = 'k', lw = 1, ls = '-', alpha = 0.8)
    
    # Align all axis labels.     
    fig.align_xlabels([ax1_x, ax2_x, ax1_xy, ax2_xy])
    fig.align_xlabels([ax3_x, ax4_x, ax3_xy, ax4_xy])
    fig.align_ylabels([ax1_y, ax3_y, ax1_xy, ax3_xy])
    fig.align_ylabels([ax2_y, ax4_y, ax2_xy, ax4_xy])
    
    # Text
    frame_text_list = []
    
    for axis, axis_label in zip(main_axes, axes_labels):
        ax_text = axis.text(0.02, 0.98, axis_label, c = 'w', fontsize = 15, transform = axis.transAxes, verticalalignment = 'top',
                            bbox={'boxstyle':'square', 'edgecolor':'k', 'facecolor':'k', 'alpha':0.25})
        ax_text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
        
        frame_text = axis.text(0.02, 0.02, frame_text, c = 'w', fontsize = 15, transform = axis.transAxes, verticalalignment = 'bottom', 
                               bbox={'boxstyle':'square', 'edgecolor':'k', 'facecolor':'k', 'alpha':0.25})
        frame_text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
        
        frame_text_list.append(frame_text)
    
    if wing_type == 'scatter':
        axis_data_list = [main_axes_data, x_projection_data, y_projection_data, scatter_mean_data, scatter_min_data, scatter_max_data]
        
    else:
        axis_data_list = [main_axes_data, x_projection_data, y_projection_data]
        
    return fig, axis_data_list, frame_text_list

def update_4_panel_tube(tube_1, tube_2, tube_3, tube_4, panel_axes_data, 
                        z_slice, frame_text_list, frame_text, wing_type = 'scatter'):
    """
    Function to update a 
    """
    tube_cross_section_dim = int(np.shape(tube_1)[0])
    position_index = int(tube_cross_section_dim/2)
    circle_slice_positions = np.array(range(0, tube_cross_section_dim))
    
    wing_type_list = ['line', 'scatter']
    
    if wing_type not in wing_type_list:
        wing_type = 'line'
    
    tube_slice_1_x = tube_1[position_index, :, z_slice]
    tube_slice_1_y = tube_1[:, position_index, z_slice]
    
    tube_slice_2_x = tube_2[position_index, :, z_slice]
    tube_slice_2_y = tube_2[:, position_index, z_slice]
    
    tube_slice_3_x = tube_3[position_index, :, z_slice]
    tube_slice_3_y = tube_3[:, position_index, z_slice]
    
    tube_slice_4_x = tube_4[position_index, :, z_slice]
    tube_slice_4_y = tube_4[:, position_index, z_slice]
        
    if wing_type == 'scatter':
        position_limits = [0, tube_cross_section_dim]
        
        tube_slice_1_x = tube_1[:,:,z_slice]
        tube_slice_1_y = tube_1[:,:,z_slice]
        
        tube_slice_2_x = tube_2[:,:,z_slice]
        tube_slice_2_y = tube_2[:,:,z_slice]
        
        tube_slice_3_x = tube_3[:,:,z_slice]
        tube_slice_3_y = tube_3[:,:,z_slice]
        
        tube_slice_4_x = tube_4[:,:,z_slice]
        tube_slice_4_y = tube_4[:,:,z_slice]
        
    main_axes = panel_axes_data[0]
    x_projection_axis = panel_axes_data[1]
    y_projection_axis = panel_axes_data[2]
    
    scatter_mean_data = []
    scatter_min_data = []
    scatter_max_data = []
    
    if wing_type == 'scatter':
        scatter_mean_data = panel_axes_data[3] #[0, 1], [x, y]
        scatter_min_data = panel_axes_data[4]
        scatter_max_data = panel_axes_data[5]
        
    scatter_axis = [scatter_mean_data, scatter_min_data, scatter_max_data]
    
    main_axis_data = [tube_1[:,:,z_slice], tube_2[:,:,z_slice], tube_3[:,:,z_slice], tube_4[:,:,z_slice]]
    x_projection_axis_data = [[circle_slice_positions, tube_slice_1_x], [circle_slice_positions, tube_slice_2_x],
                              [circle_slice_positions, tube_slice_3_x], [circle_slice_positions, tube_slice_4_x]]
    y_projection_axis_data = [[tube_slice_1_y, circle_slice_positions], [tube_slice_2_y, circle_slice_positions], 
                              [tube_slice_3_y, circle_slice_positions], [tube_slice_4_y, circle_slice_positions]]
    
    for main_ax, main_data in zip(main_axes, main_axis_data):
        main_ax.set_data(main_data)
        
    for axis_index, (x_axis_data, y_axis_data, x_axis, y_axis, s_axis_mean, s_axis_min, s_axis_max) in enumerate(zip(x_projection_axis_data, y_projection_axis_data, x_projection_axis, y_projection_axis, 
                                                                                                                     scatter_mean_data, scatter_min_data, scatter_max_data)):
        
        if wing_type == 'line': # [x_projection_data, y_projection_data]
            x_axis.set_data(*x_axis_data)
            y_axis.set_data(*y_axis_data)
        
        elif wing_type == 'scatter': # [x_projection_data, y_projection_data, scatter_mean_data, scatter_min_data, scatter_max_data]
            x_locs = x_axis_data[0]
            x_vals = x_axis_data[1]
            y_locs = y_axis_data[1]
            y_vals = y_axis_data[0]
            
            loc_data = []
            long_x_data = []
            long_y_data = []
            
            for ax_slice in range(*position_limits):
                loc_data.extend(x_locs)
                long_x_data.extend(x_vals[ax_slice, :])
                long_y_data.extend(y_vals[:, ax_slice])
                
            x_axis.set_data(loc_data, long_x_data)
            y_axis.set_data(long_y_data, loc_data)
            
            s_axis_mean[0].set_data(x_locs, np.mean(x_vals, axis = 0))
            s_axis_min[0].set_data(x_locs, np.min(x_vals, axis = 0))
            s_axis_max[0].set_data(x_locs, np.max(x_vals, axis = 0))
            
            s_axis_mean[1].set_data(np.mean(y_vals, axis = 1), y_locs)
            s_axis_min[1].set_data(np.min(y_vals, axis = 1), y_locs)
            s_axis_max[1].set_data(np.max(y_vals, axis = 1), y_locs)
    
    for text in frame_text_list:
        text.set_text('{0}'.format(frame_text))
    
    return

def save_4_panel_tube_sequence(loaded_file_list, plot_dir, t_start = time.time(), file_prefix = 'lambda', subplot_dim = 6,
                               wing_type = 'scatter', def_dist = [-1, 0], **setup_kwargs):
    """
    Function to plot a 4 panel tube sequence along the 3rd axis (taken to be z-axis) of a set of 4
    3D arrays of the same shape. 
    
    Uses setup_4_panel_tube and update_4_panel_tube to reduce total number of redraws to be done. 
    Saves using fig.canvas.print_figure(plot_path, facecolor = 'white', edgecolor = 'white').
    This reduces the time to produce the plots dramatically. 
    
    Parameters:
    ----------
    loaded_file_list: list 
        List of 4 arrays to be plotted. 
        
    plot_dir: string
        Plot output directory. 
        
    t_start: float
        Time of start. Used to inform user of time elapsed. 
        Default is current cpu clock time via time.time().
        
    file_prefix: string
        File prefix to prepend to the saved images. 
        Default is 'lambda'.
        
    subplot_dim: float or int
        Subplot dimension of the figure. Final figure is 
        2*subplot_dim by 2*subplot_dim. 
        Default is 6.
        
    wing_type: string
        Type of plot for each of the wing plots. 
        Options are 'scatter' and 'line'. 
        Default is 'scatter'.
        
    def_dist: list
        Deflection distances between the x and y values of 
        each tube and the axis of rotation. Format is [x, y]. 
        Default is [-1, 0]. 
    
    **setup_kwargs: keyword parameters 
        Parameters for setup_4_panel_tube not included by name. 
        Includes:
            subplot_dim, xy_axis_lims, histogram, bottom_label, 
            top_label, left_label, right_label, data_label. 
        See setup_4_panel_tube for more information. 
        
    Outputs:
    -------
    None
    
    Saves:
    -----
    Series of 4 panel plots. 
    
    """
    # Prefix, outdir checks. 
    file_prefix, prefix_folder = prefix_check(file_prefix)
    plot_dir = outdir_check(plot_dir)
    
    data_shape = np.shape(loaded_file_list[0])
    n_slices = data_shape[2]
    
    fig, panel_axes_data, frame_text_list = setup_4_panel_tube(data_shape = (data_shape[0], data_shape[1]), wing_type = wing_type, **setup_kwargs)
    
    for z_slice in range(n_slices):
        z_file = '{0}'.format(z_slice)
        z_file = z_file.zfill(3)
        
        plot_name = '{0}{1}.png'.format(file_prefix, z_file)
        plot_path = os.path.join(plot_dir, plot_name)
        
        # Frame shift as ref - comparison. 
        frame_shift = n_slices/2 - z_slice # Tubes are hardcoded to align at length/2, which is the same as n_slice/2. 
        
        x_shift = frame_shift * def_dist[0]/n_slices
        y_shift = frame_shift * def_dist[1]/n_slices
        frame_text = r'($\Delta$x, $\Delta$y, z) = ({0:.2f}, {1:.2f}, {2})'.format(x_shift, y_shift, z_slice)
        
        update_4_panel_tube(*loaded_file_list, panel_axes_data, z_slice, frame_text_list, frame_text, wing_type = wing_type)
        
        fig.canvas.print_figure(plot_path, facecolor = 'white', edgecolor = 'white')
        
        t_end = time.time()
        t_seconds = t_end - t_start
        t_minutes = (t_end - t_start)/60
        
        if z_slice + 1 == n_slices:
            print('Plotted slice {0} of {1}. Time elapsed {2:.2f} seconds.'.format(z_slice + 1, n_slices, t_seconds), end = '\n')
            
        else:
            print('Plotted slice {0} of {1}. Time elapsed {2:.2f} seconds.'.format(z_slice + 1, n_slices, t_seconds), end = '\r')
            
    plt.close('all')
            
    return

def plot_2d_histogram(x_data, y_data, x_label = 'X Data', y_label = 'Y Data', bins = 10, vmax = 25, 
                      x_ticks = None, x_ticklabels = None, y_ticks = None, y_ticklabels = None, 
                      cbar_label = 'Counts', image_text = None, image_title = 'Histogram'):
    """
    I want this function to take in data and produce 2D histograms.     
    
    """

    H, x_edges, y_edges = np.histogram2d(x_data, y_data, bins = bins)
    
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    plt.close('all')
    
    fig, ax = plt.subplots(figsize = (7, 6)) #Oversized in the x dimension to make room for colorbar. 
    
    im = ax.imshow(H.T, cmap = cm.m_fire, extent = extent, aspect = 'auto', origin = 'lower', vmax = vmax)
    add_colorbar(ax, im, fig, cbar_label = cbar_label, ylabel_kwargs = {'fontsize': 15}, tick_param_kwargs = {'labelsize': 12})
    
    if image_text != None:
        slice_text = ax.text(0.75, 0.95, image_text, c = 'w', fontsize = 15, transform = ax.transAxes)
        slice_text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
    
    ax.set_xlabel(x_label, fontsize = 15)
    ax.set_ylabel(y_label, fontsize = 15)
    ax.tick_params(labelsize = 12)
    
    ax.set_title(image_title, fontsize = 20)
    
    if x_ticks != None:
        ax.set_xticks(x_ticks)
    
    if y_ticks != None:
        ax.set_yticks(y_ticks)
    
    plt.tight_layout()
    plt.show()
    
    return 

def save_2d_histogram_sequence(x_data_volume, y_data_volume, image_title = 'Histogram',
                            x_label = 'X Data', y_label = 'Y Data', bins = 10, vmax = 25,
                            x_ticks = None, x_ticklabels = None, y_ticks = None, y_ticklabels = None,
                            cbar_label = 'Counts', animate = True, animate_only = False, 
                            outdir = os.getcwd(), file_name = 'histogram'):
    """
    
    """ 
    
    outdir = outdir_check(outdir)
    
    image_prefix, folder = prefix_check(file_name)
    
    t_start = time.time()
    
    plt.close('all')
    
    fig, ax = plt.subplots(figsize = (6.5, 6))
    
    n_images = np.shape(x_data_volume)[2]
    zfill_len = len('{0}'.format(n_images))
    
    x_data = x_data_volume[:,:,0].flatten()
    y_data = y_data_volume[:,:,0].flatten()
    
    H, x_edges, y_edges = np.histogram2d(x_data, y_data, bins = bins)
    
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    
    im = ax.imshow(H.T, cmap = cm.m_fire, extent = extent, aspect = 'auto', origin = 'lower', vmax = vmax)
    
    slice_text = ax.text(0.71, 0.95, 'Slice = {0}'.format(0), c = 'w', fontsize = 15, transform = ax.transAxes)
    slice_text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
    
    ax.set_xlabel(x_label, fontsize = 15)
    ax.set_ylabel(y_label, fontsize = 15)
    ax.tick_params(labelsize = 12)
    
    ax.set_title(image_title, fontsize = 20)
    
    if x_ticks != None:
        ax.set_xticks(x_ticks)
    
    if y_ticks != None:
        ax.set_yticks(y_ticks)
        
    add_colorbar(ax, im, fig, cbar_label = cbar_label, ylabel_kwargs = {'fontsize': 15}, tick_param_kwargs = {'labelsize': 12})
    plt.tight_layout()
    
    plot_num = '{0}'.format(0).zfill(zfill_len)
    plot_filename = '{0}{1}.png'.format(image_prefix, plot_num)
    plot_path = os.path.join(outdir, plot_filename)
    
    fig.canvas.print_figure(plot_path, facecolor = 'white', edgecolor = 'white')
    
    for image_number in range(1, n_images):
        
        x_data = x_data_volume[:,:,image_number].flatten()
        y_data = y_data_volume[:,:,image_number].flatten()
        
        H, x_edges, y_edges = np.histogram2d(x_data, y_data, bins = bins)
        im.set_data(H.T)
        slice_text.set_text('Slice = {0}'.format(image_number))
        
        plot_num = '{0}'.format(image_number).zfill(zfill_len)
        plot_filename = '{0}{1}.png'.format(image_prefix, plot_num)
        plot_path = os.path.join(outdir, plot_filename)
        fig.canvas.print_figure(plot_path, facecolor = 'white', edgecolor = 'white')
        
        t_end = time.time()
        t_diff = t_end - t_start
        
        if image_number + 1 < n_images:
            print('Saved image {0} of {1}. Time elapsed: {2:.2f} seconds.'.format(image_number + 1, n_images, t_diff), end = '\r')
            
        else:
            print('Saved image {0} of {1}. Time elapsed: {2:.2f} seconds.'.format(image_number + 1, n_images, t_diff))
            
    plt.close('all')
                
    if animate:
        file_name_pattern = '{0}%0{1}d.png'.format(image_prefix, zfill_len)
        search_dir = outdir
        outdir = outdir
        video_name = file_name
        
        create_animation(file_name_pattern, framerate = 10, search_dir = search_dir, outdir = outdir, 
                         video_name = video_name, video_format = 'mp4', overwrite = True)
    
    return

def save_multiple_histogram_sequences(parent_path, noise_levels = [0.00, 0.01, 0.02, 0.03], variables = ['lambda', 'theta'], animate = True):
    """
    
    """
    
    label_dict = {'lambda': r'$\lambda$', 'theta': r'$\theta$ [Degrees]', 'distance': 'Distance [Voxels]', 'difference': r'Difference [$\rho$]'}
    column_dict = {'lambda': 0, 'theta': 1, 'distance': 5, 'difference': 6}
    bin_dict = {'lambda': np.arange(0 - 0.05, 2 + 0.1, 0.1),
                'theta': np.arange(-90 - 5, 90 + 10, 10),
                'distance': np.arange(0 - 0.025, 1 + 0.05, 0.05),
                'difference': np.arange(0 - 0.025, 1 + 0.05, 0.05)}
    
    tick_dict = {'lambda': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00],
                'theta':  [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90],
                'distance': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'difference': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    
    output_file_prefix = '{0}_{1}'.format(variables[1], variables[0])
    x_data_col = column_dict[variables[0]]
    y_data_col = column_dict[variables[1]]
    
    x_label = label_dict[variables[0]]
    y_label = label_dict[variables[1]]
    x_ticks = tick_dict[variables[0]]
    y_ticks = tick_dict[variables[1]]
    
    bins = [bin_dict[variables[0]], bin_dict[variables[1]]]
    vmax = 25
    
    results_file_list = []
    image_title_list = []
    outdir_list = []
    
    for noise_level in noise_levels:
        if noise_level == 0.00:
            file_dir = os.path.join(parent_path, 'Clean')
            results_file_path = os.path.join(file_dir, 'clean_results.txt')
            image_title = 'Clean'
            
            results_file_list.append(results_file_path)
            image_title_list.append(image_title)
            outdir_list.append(os.path.join(file_dir, 'Plots', 'Histograms', output_file_prefix))
        
        else:
            file_dir = os.path.join(parent_path, 'Noise', '{0}'.format(noise_level))
            noise_results_file_path = os.path.join(file_dir, 'noise_results.txt')
            noise_image_title = 'Noise: {0}'.format(noise_level)
            
            results_file_list.append(noise_results_file_path)
            image_title_list.append(noise_image_title)
            outdir_list.append(os.path.join(file_dir, 'Plots', 'Histograms', output_file_prefix, 'Noise'))
            
            glob_pattern = os.path.join(file_dir, '*results_smooth*.txt')
            smooth_results_file_paths = glob.glob(glob_pattern)
            
            for smooth_result_file in smooth_results_file_paths:
                base_name = os.path.basename(smooth_result_file)
                base_name = base_name.split('.txt')[0]
                smooth_kernal = base_name.split('_')[-1]
                
                results_file_list.append(smooth_result_file)
                image_title_list.append('Noise: {0}, Smoothing Kernal: {1}'.format(noise_level, smooth_kernal))
                outdir_list.append(os.path.join(file_dir, 'Plots', 'Histograms', output_file_prefix, 'Smooth_{0}'.format(smooth_kernal)))
                
    for result_path, image_title, histogram_outdir in zip(results_file_list, image_title_list, outdir_list):
        result_filename = os.path.basename(result_path)
        print('Working on noise level {0}. Source file: {1}.'.format(image_title, result_filename))
        
        x_data_volume = load_flat_array(result_path, usecols = x_data_col)
        y_data_volume = load_flat_array(result_path, usecols = y_data_col)
        
        histogram_outdir = outdir_check(histogram_outdir)
        
        save_2d_histogram_sequence(x_data_volume, y_data_volume, image_title = image_title,
                                   x_label = x_label, y_label = y_label, bins = bins, vmax = vmax, 
                                   x_ticks = x_ticks, x_ticklabels = None, y_ticks = y_ticks, y_ticklabels = None,
                                   cbar_label = 'Counts', animate = animate, outdir = histogram_outdir, file_name = output_file_prefix,)
        
        print('-'*35)
        print('')
    
    return

def plot_reference_volumes(vol_list, title_list, plot_dir):
    """
    
    """ 
    
    vmin = np.min(vol_list[0])
    vmax = np.max(vol_list[0])
    
    plt.close('all')
    fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(2, 3, figsize = (18, 12))
    
    ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    
    for ax, vol, title in zip(ax_list, vol_list, title_list):
        im = ax.imshow(vol[:,:,0], cmap = cm.m_fire, origin = 'lower', vmin = vmin, vmax = vmax)
        
        ax.tick_params(labelsize = 12)
        ax.set_xlabel('X Position [Voxel]', fontsize = 15)
        ax.set_ylabel('Y Position [Voxel]', fontsize = 15)
        
        ax.set_title(title, fontsize = 20)
        
        add_colorbar(ax, im, fig, tick_param_kwargs = {'labelsize': 12})
    
    plt.tight_layout()
    plot_name = 'volume_check.png'
    plot_path = os.path.join(plot_dir, plot_name)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close('all')
    
    print('Saved volume check at {0}.'.format(plot_path))

    return

####################################################
#
# General Helper Functions.
#
####################################################

def outdir_check(outdir):
    """
    Helper function to check whether outdir conforms to expected syntax. Creates outdir if it does not already exist. 
    
    Parameters:
    ----------
    outdir: string
        Output directory for a given function. 
        
    Outputs:
    -------
    outdir: string
        Checked output directory. 
    """
    # Split outdir into components.
    drive, path = os.path.splitdrive(outdir)
    path_components = path.split(os.sep)
    
    outdir = '{0}{1}'.format(drive, os.path.sep)
    
    created_paths = []
    
    for component in path_components:
        outdir = os.path.join(outdir, '{0}'.format(component))
        outdir = os.path.normpath(outdir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
            created_paths.append(outdir)
    
    for new_path in created_paths:
        print('Created {0}'.format(new_path))
    
    return outdir

def source_check(source_dir):
    """
    Helper function to check whether a source directory conforms to expected syntax. 
    
    Parameters:
    ----------
    source_dir: string
        Source directory for a given purpose. 
        
    Output:
    ------
    source_dir: string
        source_dir with appropriate syntax for a given os. 
    
    """
    drive, path = os.path.splitdrive(source_dir)
    path_components = path.split(os.path.sep)
    
    source_dir = '{0}{1}'.format(drive, os.path.sep)
    
    for component in path_components:
        source_dir = os.path.join(source_dir, '{0}'.format(component))
        source_dir = os.path.normpath(source_dir)
    
    return source_dir 

def prefix_check(prefix = ''):
    """
    Helper function to standardize file prefixes. Returns either an empty string '' or a string with an underscore. 
    
    Parameters:
    ----------
    prefix: string
        Prefix for a given filename. If empty, returns and empty string. 
        If nonempty, checks for an underscore. 
        Default is ''.
        
    Outputs:
    -------
    file_prefix: string
        Standardized filename prefix. 
        
    folder_prefix: string
        Standardized folder prefix.
    """
    if prefix == '':
        file_prefix = ''
        folder_prefix = ''
        
    elif prefix[-1] == '_':
        file_prefix = prefix
        folder_prefix = prefix.split('_')[0]
    else: 
        file_prefix = '{0}_'.format(prefix)
        folder_prefix = prefix
        
    return file_prefix, folder_prefix

def save_flat_array(array, file_path, order = 'C', **savetxt_params):
    """
    Function to save a numpy array in the appropriate format for meshgamma to 
    parse it. 
    
    Swaps the 1st and 2nd axes of the array to ensure that X and Y axes
    are where we expect them when we plot the results. 
    
    Uses np.savetxt to save the file. 
    
    Parameters:
    ----------
    array: Numpy Array
        Array to be saved. Assumed to be 3D. 
        
    file_path: string
        Path of the file to be saved. 
        
    order: string
        Order to flatten the the data with. 
        Options are 'C', 'F', 'A', 'K'. 
        'C' is row-major, 'F' is column-major, 
        'A' is column-major if the array is Fortran 
        contiguous in memory, else row-order. 
        'K' is in the order array elements are in memory. 
        Default is 'C'. 
        
    savetxt_params: Parameters for np.savetxt. 
        Parameters for np.savetxt function. 
        Includes:
            fmt, delimiter, newline, header, footer, 
            comments, encoding. 
            
        See the np.savetxt docstring for more information. 
            
    Outputs:
    -------
    None
    
    Saves:
    -----
    file_path: text file   
    
    """
    
    # Switch 1st and 2nd axis. 
    array = np.swapaxes(array, 0, 1)
    
    flat_array = array.flatten(order = order)
    
    np.savetxt(file_path, flat_array, **savetxt_params)
    
    return

def load_flat_array(file_name, skiprows = 3, order = 'C', usecols = None):
    """
    Function to load a flattened array saved with either meshgamma or the 
    save_flat_array function. 
    
    Uses np.loadtxt. 
    
    Parameters:
    ----------
    file_name: string
        File to be loaded. 
        
    skiprows: int
        Number of rows to skip before data begins. 
        For meshgamma results, ensures that dimension header is
        not included in the final array. 
        Default is 4. 
        
    order: string
        Ordering of the flattened array. 
        Default is 'C'.
        
    usecols: int or None
        Desired data column to access. 
        If None, will load all columns. 
        Default is None. 
        
    Outputs:
    -------
    array: Numpy array
        Array composed of reshaped data values. 
        If N columns are accessed, shape will be [N, X, Y, Z]. 
    
    """
    # Load flattened array and transpose it so that each result variable
    # is a row instead of a column. 
    flat_array = np.loadtxt(file_name, skiprows = skiprows, usecols = usecols)
    flat_array = flat_array.T
    
    array_shape = [len(flat_array)]
    shape = []
    
    # Access dimension data for reshaping. 
    # These are the first 3 lines. 
    line_n = 0
    with open(file_name) as file:
        for line in file:
            if line_n < 3:
                array_shape.append(int(line))
                shape.append(int(line))
                line_n += 1
    
    # If a single column was chosen, reshape into appropriate shape and return. 
    if flat_array.ndim == 1:
        array = flat_array.reshape(tuple(shape), order = order)
        array = np.swapaxes(array, 0, 1) # Swap x and y axes. 
    
    # If multiple columns were chosen, access each and reshape into appropriate shape
    # and insert into 4D array. 
    else:
        array = np.zeros(tuple(array_shape))
        
        for var_num, var_array in enumerate(flat_array):
            var_array = np.reshape(var_array, tuple(shape), order = order)
            var_array = np.swapaxes(var_array, 0, 1)
            array[var_num] = var_array
    
    return array