import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
from collections import OrderedDict
import pickle
import glob
import time
import datetime
import copy
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf
from scipy.ndimage import gaussian_filter
import colorcet as cm

import plotting_utilities as plot_util

####################################################
#
# Logging and Syntax Check Functions.
#
####################################################

def write_log(log_name, log_dict, log_mode = 'w'):
    """
    Logging function to be used throughout a given module. 
    
    Automatically writes the UTC date as the first line. Uses datetime module. 
    
    Currently unused.
    
    Parameters:
    ----------
    log_name: string
        Complete file path and name of the log file. 
        e.g.: '/u/pboyle/work/data/wd1_general_completeness/Fixed_Test/wd1_completeness_cuts.log'
        
    log_dict: Ordered Dictionary
        Dictionary of ordered dictionaries to be written to the log file. Each key that is an 
        ordered dictionary will be written as a section, and sub keys that also correspond to
        ordered dictionaries will be written as subsections. 
        e.g.:
        
        OrderedDict([('Input Parameters',
              OrderedDict([('Catalog',
                            '/u/pboyle/work/data/wd1_general_completeness/Artstar_Error/wd1_artstar_catalog_cor.fits'),
                           ('Number of Epochs', 4),
                           ('1 Pix Position Cut', False),
                           ('Time Dictionary', None),
                           ('Use Transformed', False),
                           ('Artificial', True)])),
             ('Error Correction',
              OrderedDict([('2005_F814W',
                            OrderedDict([('Zero Point', 14.5698),
                                         ('Bright Floor', 16),
                                         ('Obs Mag', 0.0092),
                                         ('Obs Xerr', 0.135),
                                         ('Obs Yerr', 0.158)])),
                           ('2010_F160W',
                            OrderedDict([('Zero Point', 14.5698),
                                         ('Bright Floor', 16),
                                         ('Obs Mag', 0.0092),
                                         ('Obs Xerr', 0.135),
                                         ('Obs Yerr', 0.158)])),
                           ('2010_F139M',
                            OrderedDict([('Zero Point', 14.5698),
                                         ('Bright Floor', 16),
                                         ('Obs Mag', 0.0092),
                                         ('Obs Xerr', 0.135),
                                         ('Obs Yerr', 0.158)])),
                           ('2010_F125W',
                            OrderedDict([('Zero Point', 14.5698),
                                         ('Bright Floor', 16),
                                         ('Obs Mag', 0.0092),
                                         ('Obs Xerr', 0.135),
                                         ('Obs Yerr', 0.158)])),
                           ('2013_F160W',
                            OrderedDict([('Zero Point', 14.5698),
                                         ('Bright Floor', 16),
                                         ('Obs Mag', 0.0092),
                                         ('Obs Xerr', 0.135),
                                         ('Obs Yerr', 0.158)])),
                           ('2015_F160W',
                            OrderedDict([('Zero Point', 14.5698),
                                         ('Bright Floor', 16),
                                         ('Obs Mag', 0.0092),
                                         ('Obs Xerr', 0.135),
                                         ('Obs Yerr', 0.158)]))])),
             ('Velocity Calculations',
              OrderedDict([('Sources Present', 735035),
                           ('Sources Used', 444943),
                           ('Years Used',
                            '2005.485, 2010.652, 2010.652, 2010.652, 2013.199, 2015.148'),
                           ('Vx Median', -0.00039737512605765585),
                           ('Vxe Median', 0.0007045558395986246),
                           ('Vy Median', 3.44157318864376e-05),
                           ('Vye Median', 0.0007210671750637592)])),
             ('Decorator', ['*', 35])])
             
        would be written as: 
        
        Date: Wed Feb 20 22:54:53 2019
        Input Parameters
        ----------------
        Catalog           : /u/pboyle/work/data/wd1_general_completeness/Artstar_Error/wd1_artstar_catalog_cor.fits
        Number of Epochs  : 4
        1 Pix Position Cut: False
        Time Dictionary   : None
        Use Transformed   : False
        Artificial        : True
        
        ***********************************
        
        Error Correction
        ----------------
        2005_F814W: Zero Point      14.5698
                    Bright Floor    16
                    Obs Mag         0.0092
                    Obs Xerr        0.135
                    Obs Yerr        0.158
        
        2010_F160W: Zero Point      14.5698
                    Bright Floor    16
                    Obs Mag         0.0092
                    Obs Xerr        0.135
                    Obs Yerr        0.158
        
        2010_F139M: Zero Point      14.5698
                    Bright Floor    16
                    Obs Mag         0.0092
                    Obs Xerr        0.135
                    Obs Yerr        0.158
        
        2010_F125W: Zero Point      14.5698
                    Bright Floor    16
                    Obs Mag         0.0092
                    Obs Xerr        0.135
                    Obs Yerr        0.158
        
        2013_F160W: Zero Point      14.5698
                    Bright Floor    16
                    Obs Mag         0.0092
                    Obs Xerr        0.135
                    Obs Yerr        0.158
        
        2015_F160W: Zero Point      14.5698
                    Bright Floor    16
                    Obs Mag         0.0092
                    Obs Xerr        0.135
                    Obs Yerr        0.158
        
        ***********************************
        
        Velocity Calculations
        ---------------------
        Sources Present: 735035
        Sources Used   : 444943
        Years Used     : 2005.485, 2010.652, 2010.652, 2010.652, 2013.199, 2015.148
        Vx Median      : -0.00039737512605765585
        Vxe Median     : 0.0007045558395986246
        Vy Median      : 3.44157318864376e-05
        Vye Median     : 0.0007210671750637592
        
        ***********************************
        
        END OF LOG
                  
    log_mode: string
        Any mode signifier that the open() function takes in: 'w', 'r', 
        'x', 'a', etc. 
        Default is 'w'.
        
    Output:
    ------
    log_name: log file
        Log file specified above. 
    """    
    # Using with open() here to ensure the file is closed in the event of an error. 
    with open(log_name, mode = log_mode) as _log:
    
        utc_datetime = datetime.datetime.utcnow()
        
        # Get user information
        user = os.environ.get('USER')
        if type(user) == type(None):
            user = os.environ.get('USERNAME')
        
        _log.write('Date: {0}\n'.format(datetime.datetime.ctime(utc_datetime)))
        _log.write('User: {0}\n\n'.format(user))
        
        # Check for decorator
        dec_boolean = False
        
        if 'Decorator' in list(log_dict.keys()):
            dec_boolean = True
            decorator = '{0}'.format(log_dict['Decorator'][0]*log_dict['Decorator'][1])
            
        for section_name in log_dict.keys():
            section = log_dict[section_name]
            
            # Test each section to see if it's an
            # ordered dictionary. If so, go down one more level. 
            if type(section) == type(OrderedDict()):
                
                if dec_boolean:
                    _log.write('{0}\n\n'.format(decorator))
                
                _log.write('{}\n'.format(section_name))
                _log.write('{}\n'.format('-'*len(section_name)))
                
                # Get the list of subsection names. 
                subsection_name_list = list(section.keys())
                
                # Find longest subsection name and vertically align entries accordingly. 
                section_name_len = 0
                section_name_len = np.max([len(name) for name in subsection_name_list if section_name_len < len(name)])
                
                for subsection_name in subsection_name_list:
                    subsection = section[subsection_name]
                    
                    # Test each subsection to to see if it's an 
                    # ordered dictionary. If so, go down one more level. 
                    if type(subsection) == type(OrderedDict()):
                        subsection_value_list = list(subsection.keys())
                        
                        # Find longest subsubsection name and vertically align entries accordingly.
                        # Calling it subsect_value_len to avoid calling it subsubsect_name_len.
                        subsect_value_len = 0
                        subsect_value_len = np.max([len(name) for name in subsection_value_list if subsect_value_len < len(name)])
                        
                        # Write subsubsection information to log file. 
                        for subsection_value in subsection.keys():
                            subsection_data = subsection[subsection_value]
                            
                            if subsection_value == subsection_value_list[0]:
                                left_side = '{s_name: <{name_len}}: '.format(s_name = subsection_name, name_len = section_name_len)
                            else:
                                left_side = '{blank: <{name_len}}  '.format(blank = '', name_len = section_name_len)
                            
                            right_side = '{ss_name: <{name_len}}     {ss_value}'.format(ss_name = subsection_value, 
                                                                                        name_len = subsect_value_len, 
                                                                                        ss_value = subsection_data)
                                
                            _log.write('{0}{1}\n'.format(left_side, right_side))
                        
                        if subsection_name != subsection_name_list[-1]:
                            _log.write('\n')
                    
                    # If the subsection is not an ordered dictionary, write the 
                    # information to the log file 
                    else:
                        _log.write('{s_name: <{name_len}}: {s_value}\n'.format(s_name = subsection_name, 
                                                                               name_len = section_name_len, 
                                                                               s_value = subsection))
                        
                _log.write('\n')
                
        if dec_boolean:
            _log.write('{0}\n\n'.format(decorator))
        
        # Indicate the end of the log in the log file. 
        _log.write('END OF LOG')
                
    print('Log file written to {0}'.format(log_name))
                
    return

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
    
    # Collect each path component and check if each exists. 
    # Create paths if they do not. 
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
    Helper function to check whether a source directory conforms to expected for a 
    given operating system. 
    
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

def prefix_check(prefix = '', return_folder = False):
    """
    Helper function to standardize file prefixes. Returns either an empty string '' 
    or a string with an underscore '{prefix}_'. 
    
    Parameters:
    ----------
    prefix: string
        Prefix for a given filename. If empty, returns and empty string. 
        If nonempty, checks for an underscore. 
        Default is ''.
        
    return_folder: Boolean
        If True, will return a standardized prefix and folder. 
        Default is False. 
        
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
        
    if return_folder:
        return file_prefix, folder_prefix
    
    else:
        return file_prefix

####################################################
#
# Helper functions for opening and closing archives.
#
####################################################

def open_archive(file_name):
    """
    Helper function to open binary archive files. 
    Uses json. 
    """
    with open(file_name, 'rb') as file_archive:
        file_dict = pickle.load(file_archive)
    return file_dict

def save_archive(file_name, save_data):
    """
    Helper function to save a file as a binary archive. 
    Uses json. 
    """
    with open(file_name, 'wb') as outfile:
        pickle.dump(save_data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    return

####################################################
#
# Array file loading and saving. 
#
####################################################

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
        Default is 3. 
        
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

def save_flat_dict(data_dict, ref_image = 4, file_prefix = 'rho', order = 'C', header = '', 
                   fmt = '%.18f', comments = '', outdir = os.getcwd()):
    """
    Function to save flattened image arrays. 
    
    Parameters: 
    ----------
    data_dict: Python Dictionary
        Data dictionary to be saved. 
    
    ref_image: int
        Reference image number using 0-indexing. 
        Default is 4. 
        
    file_prefix: string
        
    """
    
    outdir = outdir_check(outdir)
    file_prefix, folder_prefix = prefix_check(file_prefix)
    
    image_array = np.squeeze(data_dict['densityAll'])
    jac_array = np.squeeze(data_dict['jacobianScanAll'])
    
    n_images = np.shape(image_array)[-1]
    
    rho_jac_list = [image_array[:,:,:,i_num]/jac_array[:,:,:,i_num] for i_num in range(n_images) if i_num != ref_image]
    rho_jac_array = np.array(rho_jac_list)
    
    xsize, ysize, zsize = np.shape(image_array[:,:,:,ref_image])
    
    ref_image = image_array[:,:,:,ref_image]
    
    header = '{0}\n{1}\n{2}'.format(xsize, ysize, zsize)
    
    ref_file_name = os.path.join(outdir, '{0}ref.txt'.format(file_prefix))
    
    save_flat_array(ref_image, ref_file_name, header = header, fmt = fmt, comments = comments, order = order)
    
    print('Saved reference file at {0}.'.format(ref_file_name))
    
    for image_num, image_array in enumerate(rho_jac_array):
        fill_num = '{0}'.format(image_num + 1)
        fill_num = fill_num.zfill(len(str(n_images)))
        file_name = os.path.join(outdir, '{0}{1}.txt'.format(file_prefix, fill_num))
        
        save_flat_array(image_array, file_name, header = header, fmt = fmt, comments = comments, order = order)
        
        print('Saved file {0} of {1}.'.format(image_num + 1, n_images - 1), end = '\r')
    
    return

def convert_array_to_list(array):
    """
    Function to transform an array of a n dimensions into a set of lists. 
    
    Parameters:
    ----------
    array: numpy array
    
    Output:
    ------
    list_array: list
        List of the positions and values of the array. 
        For a 3D array, the list array is [y, x, z, values]. 
    """
    
    # Returns y, x, z for a 3D array. Row major ordering. 
    # It doesn't have a real practical effect. 
    shape_list = np.shape(array)
    
    array_indices = np.indices(shape_list)
    
    n_dims = len(shape_list)
    n_elements = np.prod(shape_list)
    
    array_indices = np.reshape(array_indices, (n_dims, n_elements), order = 'C')
    elements = list(array.flatten(order = 'C'))
       
    list_array = [list(indices) for indices in array_indices]
    list_array.append(elements)
    
    return list_array

def get_indices_elements(array):
    """
    Same functionality as convert_array_to_list function, 
    but returns a separate arrays of coordinates and values. 
    """
    
    # Returns y, x, z for a 3D array. Row major ordering. 
    # It doesn't have a real practical effect. 
    shape_list = np.shape(array)
    
    array_indices = np.indices(shape_list)
    
    n_dims = len(shape_list)
    n_elements = np.prod(shape_list)
    
    array_indices = np.reshape(array_indices, (n_dims, n_elements), order = 'C')
    elements = array.flatten(order = 'C')
    
    return array_indices, elements

####################################################
#
# Test Volume Creation
#
####################################################

def sigmoid_array(x_0, x_vals, M_max = 1.0, m_min = 0.15, sigma = 0.8):
    """
    Helper function to smooth edge transitions into ones matching the cumulative distribution
    function of a Gausssian curve. Produces a sigmoid curve about x_0 with standard deviation of sigma.  
    Uses scipy.special.erf as erf. 
    
    The function used is:
    
                                               x_vals - x_0
        0.5 * (M_max - m_min) * (1 + erf[  --------------------  ]) + m_min
                                            sigma * np.sqrt(2) 
        
    which creates a sigmoid curve with a maximum value of M_max, minimum value of m_min, 
    and a mid value of (0.5 * (M_max - m_min)). 
    
    Parameters:
    ----------
    x_0: float or array of floats. 
        Midpoint of the sigmoid curve. 
        If an array, must match the dimension of x_vals. 
        
    x_vals: array of floats. 
        Values to calculate the sigmoid curve at. 
        
    M_max: float
        Maximum value of the sigmoid curve. 
        Default is 1.0.
    
    m_min: float
        Minimum value of the sigmoid curve. 
        Default is 0.15. 
    
    sigma: float
        Standard deviation of the underlying Gaussian distribution. 
        Default is 0.8. 
        
    Outputs:
    -------
    sigmoid_curve: numpy array
        Resulting values of the sigmoid curve. 
        Dimensions match shape of x_vals.    
    """

    sigmoid_curve = 0.5 * (M_max - m_min) * (1 + erf((x_vals - x_0)/(sigma * np.sqrt(2)))) + m_min
    
    return sigmoid_curve
    
def infinite_half_plane(x_range, x_0 = 25, shift = 0, M_max = 1, m_min = 0.15, sigma = 0.8):
    """
    Function to populate a 3D space with with M_max and m_min values on the opposite sides of
    x_0 and a sigmoid transition between them. 
    """
    
    if shift > 0:
        shift_range = np.linspace(-0.5 * shift, 0.5 * shift, len(x_range), dtype = np.float32)
    
    else:
        shift_range = np.zeros(len(x_range), dtype = np.float32)
    
    xx, xxy, shift_array = np.meshgrid(x_range, x_range, shift_range)
    
    x_0 = x_0 - shift_array
    
    sigmoid_plane = sigmoid_array(x_0, xx, M_max = M_max, m_min = m_min, sigma = sigma)
    
    return sigmoid_plane.astype(np.float(32))

def infinite_tube(tube_length, xc_shift = 0, yc_shift = 0, max_x_shift = 0, max_y_shift = 0,
                  radius = 1, epsilon = 0.25, padding = 'auto', return_distances = False, 
                  M_max = 1, m_min = 0.15, sigma = 0.8):
    """
    Function to create tubes through 3 dimensional space. 
    
    Automatically sizes 3D space using tube_length and radius, padding, and center shifts of tube. 
    Final size is [plane_size, plane_size, tube_length], where plane size is:
        plane_size = 2*(radius + padding + max_x_shift + max_y_shift)
    
    
    Parameters:
    ----------
    tube_length: int
        Number of voxels for the length of the tube.
        
    xc_shift: int or float
        X shift of the center of the tube. 
        Default is 0.
        
    yc_shift: int or float
        Y shift of the center of the tube. 
        Default is 0.
        
    max_x_shift: int or float
        Maximum X deflection of the tube. 
        Default is 0.
        
    max_y_shift: int or float
        Maximum Y defelection of the tube. 
        Default is 0.
        
    radius: int or float
        Radius of the tube. 
        Default is 5.
        
    epsilon: float
        Default is 0.25
        
    padding: string or int
        Padding added to each side of the tube in the X and Y direction. 
        If 'auto', padding will be 7 * sigma, rounded up to the nearest integer. 
        This results in ~20% of the sigmoid values at the extremum.
        
        If an int, padding will be the user specified integer. 
        Default is 'auto'. 
        
    return_distances: Boolean
        If True, will return the distance array used to calculate the interior
        of the tube. Used to perfom diagnostic checks. 
        Default is False. 
        
    m_min: float
        Minimum values in the array. 
        Default is 0.15.
        
    M_max: float
        Maximum values in the array.
        Default is 1.0.
        
    sigma: float
        Error function maximum slope, same as normal distribution sigma. 
        Default is 0.8.
        
    Outputs:
    -------
    reference_tube: numpy array
        Array of shape [plane_size, plane_size, tube_length] with
        smoothed borders via the sigmoid_array function. 
        
    comparison_tube: numpy array
        Shifted array of shape [plane_size, plane_size, tube_length] with
        smoothed borders via the sigmoid_array function. 
    """ 
    
    if padding == 'auto':
        padding = int(np.ceil(7 * sigma))
    
    x_pad = np.max([np.abs(max_x_shift), np.abs(xc_shift)])
    y_pad = np.max([np.abs(max_y_shift), np.abs(yc_shift)])
    
    axis_pad = np.max([x_pad, y_pad])

    plane_size = int(2*(radius + padding + axis_pad))
    
    x_0 = (plane_size - 1)/2
    y_0 = (plane_size - 1)/2
    
    x_ref_centers = np.array(range(0, plane_size)) - x_0
    y_ref_centers = np.array(range(0, plane_size)) - y_0
    
    z_ref_centers = np.zeros(tube_length)
    
    xx_ref, yy_ref, zz_ref = np.meshgrid(x_ref_centers, y_ref_centers, z_ref_centers)
    
    reference_distance_array = np.linalg.norm((xx_ref, yy_ref, zz_ref), axis = 0)
    
    # Negative radius and distances here because otherwise the erf_circle values are inverted. 
    reference_tube = sigmoid_array(-radius, -reference_distance_array, M_max = M_max, m_min = m_min, sigma = sigma)
    
    # If these are all zero, only return reference. 
    if xc_shift**2 + yc_shift**2 + max_x_shift**2 + max_y_shift**2 == 0:
        
        if return_distances:
            return reference_tube.astype(np.float32), reference_distance_array.astype(np.float32)
        
        else:
            return reference_tube.astype(np.float32)
    
    # Otherwise produce and return comparison tube. 
    else:
    
        x0_comp = x_0 + xc_shift
        y0_comp = y_0 + yc_shift
        
        x_shift_per_frame = max_x_shift/tube_length
        y_shift_per_frame = max_y_shift/tube_length
        
        x_comp_centers = np.array(range(0, plane_size)) - x0_comp
        y_comp_centers = np.array(range(0, plane_size)) - y0_comp
        zx_comp_centers = np.zeros(tube_length) - (np.array(range(0, tube_length)) * x_shift_per_frame)
        zy_comp_centers = np.zeros(tube_length) - (np.array(range(0, tube_length)) * y_shift_per_frame)
        
        xx_comp, yy_comp, zzx = np.meshgrid(x_comp_centers, y_comp_centers, zx_comp_centers)
        xx_comp, yy_comp, zzy = np.meshgrid(x_comp_centers, y_comp_centers, zy_comp_centers)
        
        xx_comp = xx_comp + zzx
        yy_comp = yy_comp + zzy
        
        comparison_distance_array = np.linalg.norm((xx_comp, yy_comp, zzx - zzx), axis = 0)
        
        # Negative radius and distances here because otherwise the erf_circle values are inverted. 
        comparison_tube = sigmoid_array(-radius, -comparison_distance_array, M_max = M_max, m_min = m_min, sigma = sigma)
        
        if return_distances:
            return reference_tube.astype(np.float32), reference_distance_array.astype(np.float32), comparison_tube.astype(np.float32), comparison_distance_array.astype(np.float32)
        
        else:
            return reference_tube.astype(np.float32), comparison_tube.astype(np.float32)
        
def circle(x_0 = 0, y_0 = 0, r = 1):
    """
    Function to produce a circle. 
    
    Parameters:
    ----------
    x_0: float
        X center of the circle. 
        Default is 0. 
        
    y_0: float 
        Y center of the circle. 
        Default is 0.
        
    r: float
        Radius of the cirlce. 
        Default is 1. 
        
    Outputs:
    -------
    x_values, y_values: Numpy arrays
        Value arrays for each of the x and y points
        that line on the circle. 
    """
    
    theta = np.linspace(0, 2*np.pi, 100)
    
    x_values = r*np.cos(theta) + x_0
    y_values = r*np.sin(theta) + y_0
    
    return x_values, y_values

def ellipse(x_0 = 0, y_0 = 0, r = 1, a = 1, b = 1):
    """
    Function to produce the x and y values for a given ellipse. 
    
    a and b are used to determine the relative lengths of the x and y
    values to each other. If either are 0, returns (x_0, y_0). If a = b, 
    produces a circle with radius r. 
    
    Parameters:
    ----------
    x_0: float
        X center of the ellipse. 
        Default is 0. 
        
    y_0: float
        Y center of the ellipse.
        Default is 0.
        
    r: float
        Length of the semi-major axis of the ellipse. 
        Default is 1. 
        
    a: float
        X ratio of the ellipse. 
        Default is 1. 
        
    b: float
        Y ratio of the ellipse.
        Default is 1. 
        
    Outputs:
    -------
    x_values, y_values: Numpy arrays
        X and Y values falling on the ellipse. 
    
    """
    
    theta = np.linspace(0, 2*np.pi, 100)
    
    a = np.abs(a)
    b = np.abs(b)
    
    axis_norm = np.max([a, b])
    
    if axis_norm == 0:
        return x_0, y_0
    
    else:
        a = a/axis_norm
        b = b/axis_norm
        
        x_values = r*a*np.cos(theta) + x_0
        y_values = r*b*np.sin(theta) + y_0
        
        return x_values, y_values

def ellipse_volume(length, radius = 5, shift_x = 0, shift_y = 0, n_subpix = 100, M_max = 1.0, m_min = 0.15):
    """
    Function to produce a volume holding a cylinder of 
    specified length, shifts, and maximum/minimum values. 
    
    The cylinder is distorted into an ellipse whose semi-major 
    and semi-minor axes are determined by x_shift and y_shift. 
    In particular, an x shift tilts the cylinder out of the
    X,Z plane by theta_x = np.arctan2(x_shift, length/2). 
    The ratio between the maximum radius and x radius is then
    np.tan(theta_x). 
    
    The axis of rotation for these shifts is located in the 
    Z = length/2 plane, which corresponds to halfway through
    the volume. 
    
    Volume shape is (grid_space, grid_space, length), where grid_space 
    is determined by:
        grid_space = int(4*radius + 2*np.max(np.abs([shift_x, shift_y]))) + 1
        
    If calculated grid_space < 20, sets grid_space to 20 pixels to ensure that 
    small radius cyliders are fully captured.
    
    Pixel values of the edge of the cylinder are determined by 
    volume averaging the amount of the pixel taken up by the
    edge.     
    
    Parameters:
    ----------
    length: int
        Length of the tube. 
        
    radius: int or float
        Radius of the undistorted cylinder. 
        Default is 5. 
        
    shift_x: int or float
        Max shift from x_0. 
        Default is 0.
        
    shift_y: int or float
        Max shift from y_0. 
        Default is 0.
        
    n_subpix: int
        Number of subpixels per dimension each pixel is split
        into to determine the volume averaged value. 
        Default is 100. 
        
    M_max: float
        Maximum value of the volume. 
        Default is 1.0.
        
    m_min: float
        Minimum value of the volume. 
        Default is 0.15. 
        
    Outputs:
    -------
    value_array: Numpy array
        3D array of volume values. 
    """
    
    shift_x = 2 * shift_x
    shift_y = 2 * shift_y
    
    theta_x = np.arctan2(shift_x, length/2)
    theta_y = np.arctan2(shift_y, length/2)
    
    print('(X, Y) Angle: ({0:.2f}, {1:.2f}) degrees.'.format(np.rad2deg(theta_x), np.rad2deg(theta_y)))
    
    grid_space = int(4*radius + 2*np.max(np.abs([shift_x, shift_y])))
    
    if grid_space < 20:
        grid_space = 20
    
    x_0 = grid_space/2
    y_0 = grid_space/2
    z_0 = length/2
    
    grid_space += 1
    
    rad_pad = np.sqrt(2)/2
    
    x_values = np.array(range(grid_space))
    y_values = np.array(range(grid_space))
    z_values = np.array(range(length))
    
    xx, yy, zz = np.meshgrid(x_values, y_values, z_values)
    
    x_0 = x_0 + (z_0 - zz)*np.tan(theta_x)
    y_0 = y_0 + (z_0 - zz)*np.tan(theta_y)
    
    xx = (xx - x_0)/np.cos(theta_x)
    yy = (yy - y_0)/np.cos(theta_y)
    
    dd = np.sqrt(xx**2 + yy**2) 
    cc_border = (dd >= radius - rad_pad) & (dd <= radius + rad_pad)
    cc = (dd <= radius) & ~cc_border
    border_indices = np.where(cc_border)
    
    value_array = np.ones(np.shape(dd))*m_min
    value_array[cc] = M_max
    
    xx_border = xx[border_indices]
    yy_border = yy[border_indices]
    
    xx_border_left = xx_border - 0.5
    xx_border_right = xx_border + 0.5
    yy_border_bot = yy_border - 0.5
    yy_border_top = yy_border + 0.5
    
    t_s = time.time()
    
    n_cells = len(border_indices[0])
    
    for cell_num, index in enumerate(zip(border_indices[0], border_indices[1], border_indices[2])):
        x_fine = np.linspace(xx_border_left[cell_num], xx_border_right[cell_num], n_subpix)
        y_fine = np.linspace(yy_border_bot[cell_num], yy_border_top[cell_num], n_subpix)
        
        xx_fine, yy_fine = np.meshgrid(x_fine, y_fine)
        dd_fine = np.sqrt(xx_fine**2 + yy_fine**2)
        cc_fine = dd_fine <= radius
        vv_fine = (np.sum(cc_fine)/(n_subpix * n_subpix)) * M_max
        value_array[index[0], index[1], index[2]] = vv_fine
        t_e = time.time()
        print('Border cell {0} of {1} complete. Time elapsed: {2:.2f} seconds.'.format(cell_num + 1, n_cells, t_e - t_s), end = '\r')
        
    print('Border cell {0} of {1} complete. Time elapsed: {2:.2f} seconds.'.format(n_cells, n_cells, t_e - t_s))
    print('')
    
    return value_array

def ellipse_two_volumes(length, radius = 5, shift_x_ref = 0, shift_y_ref = 0, shift_x_comp = 0, shift_y_comp = 0, n_subpix = 100, 
                        M_max = 1.0, m_min = 0.15):
    """
    Function to produce two volumes holding cylinders of 
    specified length, relative shifts, and maximum/minimum values. 
    
    The cylinder is distorted into an ellipse whose semi-major 
    and semi-minor axes are determined by the x and y shifts. 
    In particular, an x shift tilts the cylinder out of the
    X,Z plane by theta_x = np.arctan2(x_shift, length/2). 
    The ratio between the maximum radius and x radius is then
    np.tan(theta_x). 
    
    The axis of rotation for these shifts is located in the 
    Z = length/2 plane, which corresponds to halfway through
    the volume. 
    
    Volume shape is (grid_space, grid_space, length), where grid_space 
    is determined by:
        grid_space = int(4*radius + 2*np.max(np.abs([shift_x_ref, shift_y_ref, shift_x_comp, shift_y_comp]))) + 1
        
    If calculated grid_space < 20, sets grid_space to 20 pixels to ensure that 
    small radius cyliders are fully captured.
    
    Pixel values of the edge of each cylinder are determined by 
    volume averaging the amount of the pixel taken up by the
    edge.
    
    Parameters:
    ----------
    length: int
        Length of the tube. 
        
    radius: int or float
        Radius of the undistorted cylinder. 
        Default is 5. 
        
    shift_x_ref: int or float
        Max shift from x_0 in the first slice and the final slice
        of the reference cylinder. 
        Default is 0.
        
    shift_y_ref: int or float
        Max shift from y_0 in the first slice and the final slice
        of the reference cylinder. 
        Default is 0.
        
    shift_x_comp: int or float
        Max shift from x_0 in the first slice and the final slice
        of the comparison cylinder. 
        Default is 0.
        
    shift_y_comp: int or float
        Max shift from y_0 in the first slice and the final slice
        of the comparison cylinder. 
        Default is 0.
        
    n_subpix: int
        Number of subpixels per dimension each pixel is split
        into to determine the volume averaged value. 
        Default is 100. 
        
    M_max: float
        Maximum value of the volume. 
        Default is 1.0.
        
    m_min: float
        Minimum value of the volume. 
        Default is 0.15. 
        
    Outputs:
    -------
    value_array: Numpy array
        3D array of volume values. 
    """
    
    # Multiply each shift by 2 to achieve the desired relative shifts
    # to account for the axis of rotation being located at length/2. 
    shift_x_ref = 2 * shift_x_ref
    shift_y_ref = 2 * shift_y_ref
    shift_x_comp = 2 * shift_x_comp
    shift_y_comp = 2 * shift_y_comp
    
    theta_x_ref = np.arctan2(shift_x_ref, length/2)
    theta_y_ref = np.arctan2(shift_y_ref, length/2)
    
    theta_x_comp = np.arctan2(shift_x_comp, length/2)
    theta_y_comp = np.arctan2(shift_y_comp, length/2)
    
    print('Reference (X, Y) Angle: ({0:.2f}, {1:.2f}) degrees.'.format(np.rad2deg(theta_x_ref), np.rad2deg(theta_y_ref)))
    print('Comparison (X, Y) Angle: ({0:.2f}, {1:.2f}) degrees.'.format(np.rad2deg(theta_x_comp), np.rad2deg(theta_y_comp)))
    
    grid_space = int(4*radius + 2*np.max(np.abs([shift_x_ref, shift_y_ref, shift_x_comp, shift_y_comp])))
    
    if grid_space < 20:
        grid_space = 20
    
    x_0 = grid_space/2
    y_0 = grid_space/2
    z_0 = length/2
    
    grid_space += 1
    
    rad_pad = np.sqrt(2)/2
    
    x_values = np.array(range(grid_space))
    y_values = np.array(range(grid_space))
    z_values = np.array(range(length))
    
    xx, yy, zz = np.meshgrid(x_values, y_values, z_values)
    
    value_array_list = []
    id_list = ['Reference', 'Comparison']
    theta_x_list = [theta_x_ref, theta_x_comp]
    theta_y_list = [theta_y_ref, theta_y_comp]
    
    t_s = time.time()
    
    for theta_x, theta_y, array_id in zip(theta_x_list, theta_y_list, id_list):
        
        print('Working on {0} array.'.format(array_id))
        
        x_0_temp = x_0 + (z_0 - zz)*np.tan(theta_x)
        y_0_temp = y_0 + (z_0 - zz)*np.tan(theta_y)
    
        xx_temp = (xx - x_0_temp)/np.cos(theta_x)
        yy_temp = (yy - y_0_temp)/np.cos(theta_y)
    
        dd = np.sqrt(xx_temp**2 + yy_temp**2)
        cc_border = (dd >= radius - rad_pad) & (dd <= radius + rad_pad)
        cc = (dd <= radius) & ~cc_border
        border_indices = np.where(cc_border)
    
        value_array = np.ones(np.shape(dd))*m_min
        value_array[cc] = M_max
    
        xx_border = xx_temp[border_indices]
        yy_border = yy_temp[border_indices]
    
        xx_border_left = xx_border - 0.5
        xx_border_right = xx_border + 0.5
        yy_border_bot = yy_border - 0.5
        yy_border_top = yy_border + 0.5
    
        n_cells = len(border_indices[0])
    
        for cell_num, index in enumerate(zip(border_indices[0], border_indices[1], border_indices[2])):
            x_fine = np.linspace(xx_border_left[cell_num], xx_border_right[cell_num], n_subpix)
            y_fine = np.linspace(yy_border_bot[cell_num], yy_border_top[cell_num], n_subpix)
            
            xx_fine, yy_fine = np.meshgrid(x_fine, y_fine)
            dd_fine = np.sqrt(xx_fine**2 + yy_fine**2)
            cc_fine = dd_fine <= radius
            vv_fine = (np.sum(cc_fine)/(n_subpix * n_subpix)) * M_max + (np.sum(~cc_fine)/(n_subpix * n_subpix)) * m_min
            value_array[index] = vv_fine
            t_e = time.time()
            print('{0} border cell {1} of {2} complete. Time elapsed: {3:.2f} seconds.'.format(array_id, cell_num + 1, n_cells, t_e - t_s), end = '\r')
        
        print('{0} border cell {1} of {2} complete. Time elapsed: {3:.2f} seconds.'.format(array_id, n_cells, n_cells, t_e - t_s))
        print('')
              
        value_array_list.append(value_array)
    
    reference_array = value_array_list[0]
    comparison_array = value_array_list[1]
              
    return reference_array, comparison_array

####################################################
#
# Determination of Lambda
#
####################################################


def convert_HU_to_rho(data_dict, a, b, key = 'alignedImagesAll'):
    """
    Function to convert the Hounsfield Units of a slice to density. 
    
    Simple linear equation with rho = a*HU + b.
    
    Returns data_dict with new entry corresponding to rho.
    """
    
    data_dict['density'] = a*data_dict[key] + b    
    
    return data_dict

def run_meshgamma(meshgamma_path, reference_file, compared_file, output_file, rho_perc = 3.0, dta = 3.0):  
    """
    Function to run meshgamma in a shell. Does not print any outputs from meshgamma. 
    
    Uses subprocess module. 
    
    Parameters:
    ----------
    meshgamma_path: string
        Path to meshgamma. 
        
    reference_file: string
        Path to reference file. 
        
    compared_file: string
        Path to comparison file. 
        
    output_file: string
        Output filename. 
        
    rho_perc: float
        Percent difference to weight result by. 
        Default is 3.0. 
        
    dta: float
        Distance to Agreement value to weight distances by. 
        Default is 3.0. 
    """
    command_list = [meshgamma_path, reference_file, compared_file, output_file, rho_perc, dta]
    
    command = ['{0}'.format(comm) for comm in command_list]
    
    meshgamma = subprocess.Popen(command, stdout=subprocess.PIPE)
    
    meshgamma.wait()
    
    return

def run_meshgamma_files(meshgamma_path, reference_file, compare_files, outdir = os.getcwd(), prefix = 'rho', rho_perc = 3.0, dta = 3.0):
    
    outdir = outdir_check(outdir)
    
    prefix, prefix_folder = prefix_check(prefix)
    
    t_start = time.time()

    for file_num, comp_file in enumerate(compare_files):
        in_file = os.path.basename(comp_file)
        in_file = os.path.splitext(in_file)[0]
        out_file_num = in_file.split('_')[-1]
        out_file = '{0}out_{1}.txt'.format(prefix, out_file_num)
        out_path = os.path.join(outdir, out_file)
        
        t_now = time.time()
        
        print('Working on {0}. Time elapsed: {1:.2f} seconds.'.format(out_file, t_now - t_start), end = '\r')
        
        if comp_file == compare_files[-1]:
            print('Working on {0}. Time elapsed: {1:.2f} seconds.'.format(out_file, t_now - t_start))
            
        run_meshgamma(meshgamma_path, reference_file, comp_file, out_path, rho_perc, dta)
        
    t_end = time.time()
    
    print('Finished. Time elapsed: {0:.2f} seconds.'.format(t_end - t_start))
    print('Files saved in {0}'.format(outdir))
    
    return

def auto_lambda(original_reference, original_comparison, meshgamma_path, shape = 'tube', wing_type = 'scatter', 
                noise_loc = 0.0, noise_level_list = [0.00, 0.01, 0.02, 0.03], rho_perc = 10.0, dta = 0.5, 
                seq_plot_limits = None, seq_plot = ['lambda'], smoothing_only = False, smooth_kernal = 0.5,
                check_plots = True, x_shifts = [1, 0], y_shifts = [0, 0], animate = True, animate_only = False, 
                parent_dir = os.getcwd()):
    """
    Function to produce a meshgamma results file with specified gaussian noise and smoothing.
    
    File structure created by this function with default values is as follows:


                                               --> Clean*-->| --> Plots
                                               |            | txt files
                                               |
        parent_dir --> rhoperc_10.0_dta_0.5 -->|            |--> 0.01 --> | --> Plots
                                               |            |             | txt files
                                               |            |
                                               --> Noise -->|--> 0.02 --> | --> Plots
                                                            |             | txt files
                                                            |
                                                            |--> 0.03 --> | --> Plots
                                                                          | txt files
                                                                          
        * Produced for noise level 0.00
        ** --> indicates will create a folder if one does not already exist. 
        
    Parameters:
    ----------
    original_reference: numpy array
        Reference array to compare against. 
        
    original_comparison: numpy array
        Comparison array. 
        
    meshgamma_path: string
        Path to meshgamma.exe. 
        
    shape: string
        Shape of the test distribution as produced by infinite_half_plane
        or infinite_tube functions. 
        
        TODO: UPDATE FUNCTION TO HAVE 'clinical' OPTION. 
        
        Default is 'shelf'.
        
    wing_type: string
        Wing plot type for tube plots. 
        Default is 'line'.
        
    noise_loc: float
        Value about which gaussian noise is calculated. 
        Default is 0. 
        
    noise_level_list: list of float
        Noise levels, in percentage, to apply to both the reference and comparison images. 
        Default is [0.00, 0.01, 0.02, 0.03].
        
    rho_perc: float
        Value percentage criterion, the weighting factor for value differences. 
        Default is 3.0.
        
    dta: float
        Voxel distance to agreement criterion, weighting factor for physical distance. 
        Default is 1.0. 
        
    smoothing_only: Boolean
        If True, will only plot and animate the smoothed plots.
        Default is True. 
        
    smooth_kernal: float
        Gaussian smoothing kernal size. 
        Default is 0.5.
        
    check_plots: Boolean
        If True, will plot a check plot of the original and reference distribution. 
        Default is True. 
        
    animate: Boolean
        If True, will plot and animate through the slices of the meshgamma results distributions. 
        This is the most time consuming portion of the function. 
        Default is True. 
        
    parent_dir: string
        Parent directory that will be used to save all folders and files into. 
        Eg: "\Meshgamma\Tubes\Radius_5\Shift_2_Sigma_0.8\"
        Default is os.getcwd()
        
    Outputs:
    -------
    text files:
        Reference, comparison, and results txt files for each noise level. 
        Smoothed and unsmoothed versions saved, if applicable. 
        
    """
    
    # Make parent directory. 
    parent_dir = os.path.join(parent_dir, 'rhoperc_{0}_dta_{1}'.format(rho_perc, dta))
    parent_dir = outdir_check(parent_dir)
    
    # Get shape of arrays. 
    array_shape = np.shape(original_reference)
        
    # header for meshgamma files. 
    header = '{0}\n{1}\n{2}'.format(*array_shape)
    
    # Set just_animate boolean. 
    just_animate = False
    if animate and animate_only:
        just_animate = True
    
    results_file_paths = []
    
    # make info dict
    info_dict = {}
    for noise_level in noise_level_list:
        info_dict['{0}'.format(noise_level)] = {}
        if noise_level == 0:
            data_title = 'clean'
            output_dir = os.path.join(parent_dir, 'Clean')
            output_dir = outdir_check(output_dir)
            
        else:
            data_title = 'noise'
            output_dir = os.path.join(parent_dir, 'Noise', '{0}'.format(noise_level))
            output_dir = outdir_check(output_dir)
            
        output_plot_dir = os.path.join(output_dir, 'Plots')
        output_plot_dir = outdir_check(output_plot_dir)
        
        reference_gaussian_noise = np.random.normal(loc = noise_loc, scale = noise_level, size = array_shape)
        comparison_gaussian_noise = np.random.normal(loc = noise_loc, scale = noise_level, size = array_shape)
        
        reference_array = original_reference + reference_gaussian_noise
        comparison_array = original_comparison + comparison_gaussian_noise
        
        smooth_reference_array = gaussian_filter(reference_array, sigma = smooth_kernal)
        smooth_comparison_array = gaussian_filter(comparison_array, sigma = smooth_kernal)
        data_dict = OrderedDict()
        
        if data_title == 'clean':
            data_dict = {'reference': reference_array, 'comparison': comparison_array}
            
        else:
            data_dict = {'reference': reference_array, 'comparison': comparison_array, 
                         'smooth_reference': smooth_reference_array, 'smooth_comparison': smooth_comparison_array}
            
        info_dict['{0}'.format(noise_level)] = {'title': data_title, 'data': data_dict, 'plot_dir': output_plot_dir, 'outdir': output_dir}
    
    t_start = time.time()
    for data_key in list(info_dict.keys()):
        print('')
        print('Working on noise level {0}.'.format(data_key))
        print('-'*35)
        
        check_dict = info_dict[data_key]
        check_dict_data = info_dict[data_key]['data']
        data_title = info_dict[data_key]['title']
        
        if check_plots:
            plot_rows = int(len(check_dict_data.keys())/2)
            
            if shape == 'shelf':
                plot_util.plot_shelf_check_plot(check_dict, data_key, plot_rows = plot_rows, original_reference = original_reference, smooth_kernal = smooth_kernal)
                
            elif shape == 'tube':
                plot_util.plot_tube_check_plot(check_dict, data_key, plot_rows = plot_rows, original_reference = original_reference, smooth_kernal = smooth_kernal)
            
            if plot_rows > 1:
                plot_name = '{0}_check_plot_{1}.png'.format(check_dict['title'], smooth_kernal)
                
            else:
                plot_name = '{0}_check_plot.png'.format(check_dict['title'])
                
            plot_path = os.path.join(check_dict['plot_dir'], plot_name)
            plt.savefig(plot_path, bbox_inches = 'tight')
            
            plt.close('all')
        
        reference_file_name = '{0}_ref.txt'.format(check_dict['title'])
        comparison_file_name = '{0}_comp.txt'.format(check_dict['title'])
        reference_file_path = os.path.join(check_dict['outdir'], reference_file_name)
        comparison_file_path = os.path.join(check_dict['outdir'], comparison_file_name)
        
        if not just_animate:
            if not os.path.exists(comparison_file_path):
                save_flat_array(check_dict_data['reference'], reference_file_path, fmt = '%.6f', header = header, comments = '')
                save_flat_array(check_dict_data['comparison'], comparison_file_path, fmt = '%.6f', header = header, comments = '')
                
                print('Saved reference file to {0}.'.format(reference_file_path))
                print('Saved comparison file to {0}.'.format(comparison_file_path))
        
        if data_title == 'noise':
            smooth_ref_file_name = '{0}_ref_smooth_{1}.txt'.format(check_dict['title'], smooth_kernal)
            smooth_comp_file_name = '{0}_comp_smooth_{1}.txt'.format(check_dict['title'], smooth_kernal)
            
            smooth_ref_file_path = os.path.join(check_dict['outdir'], smooth_ref_file_name)
            smooth_comp_file_path = os.path.join(check_dict['outdir'], smooth_comp_file_name)
            
            if not just_animate:
                save_flat_array(check_dict_data['smooth_reference'], smooth_ref_file_path, fmt = '%.6f', header = header, comments = '')
                save_flat_array(check_dict_data['smooth_comparison'], smooth_comp_file_path, fmt = '%.6f', header = header, comments = '')
                
                print('Saved smoothed reference file to {0}.'.format(reference_file_path))
                print('Saved smoothed comparison file to {0}.'.format(comparison_file_path))
        
        # Calculate Lambda
        results_file_name = '{0}_results.txt'.format(check_dict['title'])
        results_file_path = os.path.join(check_dict['outdir'], results_file_name)
        info_dict[data_key]['results_file_path'] = results_file_path
        
        if not just_animate:
            if not os.path.exists(results_file_path):
                print('Calculating Lambdas.')
                run_meshgamma(meshgamma_path, reference_file_path, comparison_file_path, results_file_path, rho_perc = rho_perc, dta = dta)
                print('Saved results file to {0}.'.format(results_file_path))
                
            else:
                print('Clean results files found, continuing to noise calculations.')
        
        if data_title == 'noise':
            smooth_results_file_name = '{0}_results_smooth_{1}.txt'.format(check_dict['title'], smooth_kernal)
            smooth_results_file_path = os.path.join(check_dict['outdir'], smooth_results_file_name)
            info_dict[data_key]['smooth_results_file_path'] = smooth_results_file_path
            
            if not just_animate:
                print('Calculating Lambdas.')
                run_meshgamma(meshgamma_path, smooth_ref_file_path, smooth_comp_file_path, smooth_results_file_path, rho_perc = rho_perc, dta = dta)
                print('Saved smoothed results file to {0}.'.format(smooth_results_file_path))
            
                print('Lambdas calculated.')
        
        if shape == 'shelf' and animate:
            if data_title == 'clean':
                print('Plotting results and creating video.')
                clean_meshgamma_results_array = load_flat_array(results_file_path)
                
                lambda_results = clean_meshgamma_results_array[0]
                theta_results = clean_meshgamma_results_array[1]
                xyz_results = clean_meshgamma_results_array[2:5]
                dist_results = clean_meshgamma_results_array[6]
                delta_results = clean_meshgamma_results_array[7]
                
                cmap = cm.m_fire
                origin = 'lower'
                vmin = 0
                vmax = 2
                axis_labelsize = 20
                axis_textsize = 15
                axis_text_loc = [80, 95]
                axis_textstroke = True
                clean_plot_dir = os.path.join(check_dict['plot_dir'])
                clean_plot_prefix = '{0}_lambda'.format(check_dict['title'])
                
                plot_util.plot_shelf_sequence(lambda_results, cmap = cmap, origin = origin, vmin = vmin, vmax = vmax, 
                                              axis_labelsize = axis_labelsize, axis_textsize = axis_textsize, axis_text_loc = axis_text_loc, 
                                              axis_textstroke = axis_textstroke, plot_dir = clean_plot_dir, prefix = clean_plot_prefix)
                
                print('')
                
                file_name_pattern = '{0}_%3d.png'.format(clean_plot_prefix)
                framerate = 10
                search_dir = clean_plot_dir
                outdir = clean_plot_dir
                video_name = '{0}_lambda'.format(check_dict['title'])
                video_format = 'mp4'
                overwrite = True
                
                plot_util.create_animation(file_name_pattern, framerate = framerate, search_dir = search_dir, outdir = outdir, 
                                           video_name = video_name, video_format = video_format, overwrite = overwrite)
                t_end = time.time()
                print('Clean level complete. Time elapsed: {0:.2f} minutes.'.format((t_end - t_start)/60))
                print('-'*35)
                print('')
            
            if data_title == 'noise':
                print('Plotting smooth results and creating video.')
                lambda_results = load_flat_array(smooth_results_file_path, usecols = 0)
                
                cmap = cm.m_fire
                origin = 'lower'
                vmin = 0
                vmax = 2
                axis_labelsize = 20
                axis_textsize = 15
                axis_text_loc = [80, 95]
                axis_textstroke = True
                smooth_plot_dir = os.path.join(check_dict['plot_dir'], 'Smoothed')
                smooth_plot_prefix = '{0}_lambda_Smooth'.format(check_dict['title'])
                
                plot_util.plot_shelf_sequence(lambda_results, cmap = cmap, origin = origin, vmin = vmin, vmax = vmax, 
                                              axis_labelsize = axis_labelsize, axis_textsize = axis_textsize, axis_text_loc = axis_text_loc, 
                                              axis_textstroke = axis_textstroke, plot_dir = smooth_plot_dir, prefix = smooth_plot_prefix)
                
                file_name_pattern = '{0}_%3d.png'.format(smooth_plot_prefix)
                framerate = 10
                search_dir = smooth_plot_dir
                outdir = smooth_plot_dir
                video_name = '{0}_lambda_smooth'.format(check_dict['title'])
                video_format = 'mp4'
                overwrite = True
                
                plot_util.create_animation(file_name_pattern, framerate = framerate, search_dir = search_dir, outdir = outdir, 
                                           video_name = video_name, video_format = video_format, overwrite = overwrite)
                
                if not smoothing_only:
                    print('')
                    print('Plotting unsmooth results and creating video.')
                    
                    lambda_results = load_flat_array(results_file_path, usecols = 0)
                    
                    cmap = cm.m_fire
                    origin = 'lower'
                    vmin = 0
                    vmax = 2
                    axis_labelsize = 20
                    axis_textsize = 15
                    axis_text_loc = [80, 95]
                    axis_textstroke = True
                    unsmooth_plot_dir = os.path.join(check_dict['plot_dir'], 'Unsmoothed')
                    unsmooth_plot_prefix = '{0}_lambda_unsmooth'.format(check_dict['title'])
                    
                    plot_util.plot_shelf_sequence(lambda_results, cmap = cmap, origin = origin, vmin = vmin, vmax = vmax, 
                                                  axis_labelsize = axis_labelsize, axis_textsize = axis_textsize, axis_text_loc = axis_text_loc, 
                                                  axis_textstroke = axis_textstroke, plot_dir = unsmooth_plot_dir, prefix = unsmooth_plot_prefix)
                    
                    file_name_pattern = '{0}_%3d.png'.format(unsmooth_plot_prefix)
                    framerate = 10
                    search_dir = unsmooth_plot_dir
                    outdir = unsmooth_plot_dir
                    video_name = '{0}_lambda_unsmooth'.format(check_dict['title'])
                    video_format = 'mp4'
                    overwrite = True
                    
                    plot_util.create_animation(file_name_pattern, framerate = framerate, search_dir = search_dir, outdir = outdir, 
                                               video_name = video_name, video_format = video_format, overwrite = overwrite)
                    
                t_end = time.time()
                print('Noise level {0} complete. Time elapsed: {1:.2f} minutes.'.format(data_key, (t_end - t_start)/60))
                print('-'*35)
                print('')
    
    if shape == 'tube' and animate:
        if len(noise_level_list) != 4:
            return
        else:
            if type(seq_plot) == type('string'):
                seq_plot = [seq_plot]
            
            for seq_type in seq_plot:
                seq_dict = results_file_dict[seq_type]
                seq_col = seq_dict['col']
                data_label = seq_dict['data_label']
                cmap = seq_dict['cmap']
                
                if type(seq_plot_limits) != type(None): 
                    plot_lims = seq_plot_lims
                    plot_ticks = np.round(np.linspace(*plot_lims, 3), decimals = 2)
                    
                else: 
                    plot_lims = seq_dict['plot_lims']
                    plot_ticks = seq_dict['plot_ticks']
                
                max_diffs = [2*x_shifts[1] - 2*x_shifts[0], 2*y_shifts[1] - 2*y_shifts[0]]
                
                print('')
                print('Plotting {0} tube sequences.'.format(seq_type))
                print('-'*35)
                
                noise_combination = '{0}_{1}_{2}_{3}'.format(*noise_level_list)
                
                smooth_plot_dir = os.path.join(parent_dir, 'Comparisons', 'Smooth_{0}'.format(smooth_kernal), wing_type, seq_type)
                smooth_plot_dir = outdir_check(smooth_plot_dir)
                
                unsmooth_plot_dir = os.path.join(parent_dir, 'Comparisons', 'Unsmooth', wing_type, seq_type)
                unsmooth_plot_dir = outdir_check(unsmooth_plot_dir)
                
                clean_file = info_dict['{0}'.format(noise_level_list[0])]['results_file_path']
                
                unsmooth_file_list = [clean_file]
                smooth_file_list = [clean_file]
                
                for noise_l in noise_level_list[1:]:
                    unsmooth_file_list.append(info_dict['{0}'.format(noise_l)]['results_file_path'])
                    smooth_file_list.append(info_dict['{0}'.format(noise_l)]['smooth_results_file_path'])
                
                print('Loading results.')
                smooth_results = [load_flat_array(file_path, skiprows = 4, usecols = seq_col) for file_path in smooth_file_list]
                smooth_data = [result for result in smooth_results]
                
                if not smoothing_only:
                    unsmooth_results = [load_flat_array(file_path, skiprows = 4, usecols = seq_col) for file_path in unsmooth_file_list]
                    unsmooth_data = [result for result in unsmooth_results]
                    
                    print('Plotting usmoothed sequence.')
                    unsmooth_prefix = '{0}_unsmooth'.format(seq_type)
                    axes_labels = ['Clean', 'Noise = {0:.2f}\nUnsmooth'.format(noise_level_list[1]), 
                                   'Noise = {0:.2f}\nUnsmooth'.format(noise_level_list[2]), 
                                   'Noise = {0:.2f}\nUnsmooth'.format(noise_level_list[3])]
                    plot_util.save_4_panel_tube_sequence(unsmooth_data, unsmooth_plot_dir, t_start = t_start, file_prefix = unsmooth_prefix, wing_type = wing_type, max_diffs = max_diffs,
                                                         xy_axis_lims = plot_lims, xy_axis_ticklist = plot_ticks, data_label = data_label, axes_labels = axes_labels, cmap = cmap)
                
                print('Plotting smoothed sequence.')
                smooth_prefix = '{0}_smooth_{1}'.format(seq_type, smooth_kernal)
                axes_labels = ['Clean', 'Noise = {0:.2f}\nSmooth = {1}'.format(noise_level_list[1], smooth_kernal), 
                               'Noise = {0:.2f}\nSmooth = {1}'.format(noise_level_list[2], smooth_kernal), 
                               'Noise = {0:.2f}\nSmooth = {1}'.format(noise_level_list[3], smooth_kernal)]
                plot_util.save_4_panel_tube_sequence(smooth_data, smooth_plot_dir, t_start = t_start, file_prefix = smooth_prefix, wing_type = wing_type, max_diffs = max_diffs,
                                                     xy_axis_lims = plot_lims, xy_axis_ticklist = plot_ticks, data_label = data_label, axes_labels = axes_labels, cmap = cmap)
                
                plt.close('all')
                
                print('')
                print('Saving videos.')
                print('-'*35)
                
                if not smoothing_only: 
                    file_name_pattern = '{0}_unsmooth_%3d.png'.format(seq_type)
                    framerate = 10
                    search_dir = unsmooth_plot_dir
                    outdir = unsmooth_plot_dir
                    video_name = '{0}_unsmooth'.format(seq_type)
                    video_format = 'mp4'
                    overwrite = True
                    
                    plot_util.create_animation(file_name_pattern, framerate = framerate, search_dir = search_dir, outdir = outdir, 
                                               video_name = video_name, video_format = video_format, overwrite = overwrite)
                
                file_name_pattern = '{0}_smooth_{1}_%3d.png'.format(seq_type, smooth_kernal)
                framerate = 10
                search_dir = smooth_plot_dir
                outdir = smooth_plot_dir
                video_name = '{0}_smooth_{1}'.format(seq_type, smooth_kernal)
                video_format = 'mp4'
                overwrite = True
                
                plot_util.create_animation(file_name_pattern, framerate = framerate, search_dir = search_dir, outdir = outdir, 
                                           video_name = video_name, video_format = video_format, overwrite = overwrite)
            
    return

def tube_auto_lambda(length, meshgamma_path, radius = 5, x_shifts = [1, 0], y_shifts = [0, 0], n_subpix = 100, vol_sigma = 0.8, M_max = 1.0, m_min = 0.15, 
                     wing_type = 'scatter', noise_loc = 0, noise_level_list = [0.00, 0.01, 0.02, 0.03], rho_perc = 10.0, dta = 0.5, seq_plot_limits = None, 
                     seq_plot = ['lambda'], smoothing_only = False, smooth_kernals = [0.5], check_plots = True, animate = True, animate_only = False, 
                     base_dir = os.getcwd()):
    """
    
    Function to calculate lambdas for a given set of parameters. Can determine multiple noise levels and smoothing kernals, if desired. 
    
    Parameters:
    ----------
    length: int
        Length of the tube volumes in voxels. 
        
    meshgamma_path: string
        Path to the meshgamma executable. 
        
    radius: float or int
        Radius of the tube. 
        Default is 5. 
        
    x_shifts: list of float or int
        List of the x shifts of the centers of each tube. Format is [Reference, Comparison]. 
        Default is [1, 0].
        
    y_shifts: list of float or int
        List of the y shifts of the centers of each tube. Format is [Reference, Comparison].
        Default is [0, 0].
        
    n_subpix: int
        Number of subpixels to split the x and y dimensions of each border voxel into before performing 
        volume averaging. With n_subpix = 100, each voxel is value is averaged over 10000 subvoxels. 
        Default is 100. 
        
    vol_sigma: float
        Sigma value for the gaussian blurring of the volume averaged tubes. 
        Default is 0.8. 
        
    M_max: float
        Maximum value of the unaltered volumes. 
        Default is 1.0. 
        
    m_min: float
        Minimum value of the unaltered volumes. 
        Default is 0.15. 
    
    """
    
    # Create directory structure.
    parent_dir = os.path.join(base_dir, 'Tubes', 'Radius_{0}'.format(radius), 'Length_{0}'.format(length), 
                              'Shift_XRYR_{0}_{1}_XCYC_{2}_{3}_Sigma_{4}'.format(x_shifts[0], y_shifts[0], x_shifts[1], y_shifts[1], vol_sigma))
    
    parent_dir = outdir_check(parent_dir)
    
    shift_x_ref = x_shifts[0]
    shift_y_ref = y_shifts[0]
    shift_x_comp = x_shifts[1]
    shift_y_comp = y_shifts[1]
    
    reference_volume, comparison_volume = ellipse_two_volumes(length, radius = radius, shift_x_ref = shift_x_ref, shift_y_ref = shift_y_ref,
                                                              shift_x_comp = shift_x_comp, shift_y_comp = shift_y_comp, n_subpix = n_subpix, M_max = M_max, m_min = m_min)
    difference_volume = reference_volume - comparison_volume
    
    filtered_reference = gaussian_filter(reference_volume, sigma = vol_sigma)
    filtered_comparison = gaussian_filter(comparison_volume, sigma = vol_sigma)
    
    filtered_difference = filtered_reference - filtered_comparison
    
    vol_list = [reference_volume, comparison_volume, difference_volume, filtered_reference, filtered_comparison, filtered_difference]
    
    title_list = ['Reference', 'Comparison', 'Difference', 
                  r'Filtered Reference [$\sigma = ${0}]'.format(vol_sigma), 'Filtered Comparison [$\sigma = ${0}]'.format(vol_sigma), 'Filtered Difference [$\sigma = ${0}]'.format(vol_sigma)]
    
    plot_util.plot_reference_volumes(vol_list, title_list, parent_dir)
    
    if type(smooth_kernals) != type([0.5]):
        smooth_kernals = list(smooth_kernals)
        
    if type(seq_plot) == type('string'):
        seq_plot = [seq_plot]
    
    for smooth_index, smooth_kernal in enumerate(smooth_kernals):
        print('')
        print('*'*35)
        print('Working on smoothing kernal {0}.'.format(smooth_kernal))
        print('*'*35)
        print('')
        if smooth_index > 0:
            smoothing_only = True
        else:
            smoothing_only = False
            
        auto_lambda(filtered_reference, filtered_comparison, meshgamma_path, shape = 'tube', wing_type = wing_type, seq_plot = seq_plot,
                    noise_loc = noise_loc, noise_level_list = noise_level_list, rho_perc = rho_perc, seq_plot_limits = seq_plot_limits, 
                    dta = dta, smoothing_only = smoothing_only, smooth_kernal = smooth_kernal, x_shifts = x_shifts, y_shifts = y_shifts,
                    check_plots = check_plots, animate = animate, parent_dir = parent_dir)
        print('')
        print('*'*35)
        print('Completed smoothing kernal {0}.'.format(smooth_kernal))
        print('*'*35)
    
    return

####################################################
#
# Lambda Results File Dictionary.
#
####################################################

results_file_dict = {'lambda':{'col':0, 'plot_lims':[-0.1, 2.1], 'plot_ticks': [0, 1, 2], 'data_label':r'$\lambda$', 'cmap': cm.m_fire}, 
                     'theta':{'col':1, 'plot_lims':[-100, 100], 'plot_ticks': [-90, 0, 90], 'data_label':r'$\theta$ [Deg]', 'cmap': cm.m_coolwarm},
                     'distance':{'col':5, 'plot_lims':[-0.05, 1.05], 'plot_ticks': [0, 0.5, 1],'data_label':r'$\Delta\bar{X}$ [Voxels]', 'cmap': cm.m_fire},
                     'difference':{'col':6, 'plot_lims':[-1.05, 1.05], 'plot_ticks': [-1, 0, 1],'data_label':r'$\Delta\rho$ [AU]', 'cmap': cm.m_coolwarm}}