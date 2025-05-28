import numpy as np
import scipy
import math

def construct_2D_profilometer_image(txt_file, data_start_line_num=1, scale_factor=None, subtract_mean=True, ccw_90_deg_rotations=2, flip_lr=True):
    # Initialize lists for x, y, and z values
    coordinates = []
    z_values = []
    # Read file line by line
    with open(txt_file, 'r') as file:
        for line_num, line in enumerate(file):
            if line_num >= data_start_line_num:
                # Split each line into columns by whitespace
                x_str, y_str, z_str = line.split()
                x = float(x_str)
                y = float(y_str)
                # Interpret '***' as NaN for z values
                z = float(z_str) if z_str != '***' else np.nan
                
                # Append coordinate pairs and z values
                coordinates.append((x, y))
                z_values.append(z)
    # Extract x and y coordinates separately and find unique sorted values
    x_coords, y_coords = zip(*coordinates)
    x_unique = np.sort(np.unique(x_coords))
    y_unique = np.sort(np.unique(y_coords))
    # Create an empty 2D array for the grid, filled with NaN
    array_2d = np.full((len(y_unique), len(x_unique)), np.nan)
    # Use np.searchsorted to efficiently find indices in the sorted unique x and y arrays
    x_indices = np.searchsorted(x_unique, x_coords)
    y_indices = np.searchsorted(y_unique, y_coords)
    # Populate the 2D array with z values using the precomputed indices
    for x_idx, y_idx, z in zip(x_indices, y_indices, z_values):
        array_2d[y_idx, x_idx] = z
    if subtract_mean:
        array_2d -= np.nanmean(array_2d)
    if scale_factor:
        array_2d *= scale_factor
    if ccw_90_deg_rotations:
        array_2d = np.rot90(array_2d, k=ccw_90_deg_rotations, axes=(0, 1))
    if flip_lr:
        array_2d = np.fliplr(array_2d)
    return array_2d

##################### OLD FUNCTIONS #####################

# def construct_2D_profilometer_image(txt_file, txt_data_start_line_num=11, interp_2d=True, interp_method='linear', trace_sep_angle=20., interp_len=None, scale_factor=None, subtract_mean=True, normalize_center=True):
#     """
#     Constructs a 2D image based on a text file containing profilometer data. Each column in the data file correspond to a single trace.
    
#     txt_file: the name of the file to read
#     txt_data_start_line_num (int): the row number in txt_file that the data starts
#     interp_2d (bool): If True, the space in-between traces will be interpolated over. If False, the traces will be placed and left on a NaN array.
#     interp_method (str): The interpolation method used.
#     trace_sep_angle (float): The rotation angle in degrees between adjacent traces
#     interp_len (int): If given, each trace will be interpolated to the length given. Saves a lot of time before interpolating between traces.
#     scale_factor (float or int): multiplies the final 2D image by a factor to convert to units of your choice.
#     subtract_mean (bool): If True, the mean of the final 2D profilometer image will be subtracted from the array.
    
#     The first trace (which corresponds to 0 degrees) runs left to right horizontally through the center of the image. Sequential traces are placed counterclockwise
#     """
#     prof_trace_array = read_profilometer_txt_file(txt_file, data_start_line_num=txt_data_start_line_num)
#     if interp_len is not None:
#         prof_trace_array = interpolate_prof_trace_array(prof_trace_array, interp_len)
#     if normalize_center:
#         prof_trace_array = normalize_prof_trace_array(prof_trace_array)
#     prof_image = arrange_traces_on_image(prof_trace_array, trace_sep_angle=trace_sep_angle)
#     if interp_2d:
#         interp_prof_image = interpolate_prof_image(prof_image, interp_method=interp_method)
#     else:
#         interp_prof_image = prof_image
#     if scale_factor:
#         interp_prof_image *= scale_factor
#     if subtract_mean:
#         interp_prof_image -= np.nanmean(interp_prof_image)
#     return interp_prof_image

# def read_profilometer_txt_file(file_name, data_start_line_num=11):
#     """
#     Return N x M array where N is the number of 1D traces and M is the number of data points in each trace. Figure is in units of txt file.
#     data_start_line_num is the row number in the text file that the data starts.
#     """
#     data_line_container = []
#     with open(file_name, 'r') as f:
#         for line_num, raw_line in enumerate(f):
#             if line_num >= data_start_line_num:
#                 split_strip_line = raw_line.strip().split()
#                 if not split_strip_line:
#                     break
#                 line_array = np.array(split_strip_line[1:], dtype=np.float64)
#                 data_line_container.append(line_array)
#     prof_trace_array = np.stack(data_line_container, axis=0).T
#     return prof_trace_array

# def interpolate_prof_trace_array(prof_trace_array, interp_len):
#     """
#     Return an N x interp_len array where N is the number of 1D traces. Interpolation method is linear.
#     """
#     interp_prof_trace_array = np.zeros((prof_trace_array.shape[0], interp_len))
#     for i in range(prof_trace_array.shape[0]):
#         profile = prof_trace_array[i]
#         original_prof_idxs = np.arange(len(profile))
#         new_prof_idxs = np.linspace(0, len(profile)-1, interp_len)
#         interp_profile = np.interp(new_prof_idxs, original_prof_idxs, profile)
#         interp_prof_trace_array[i] = interp_profile
#     return interp_prof_trace_array

# def normalize_prof_trace_array(prof_trace_array):
#     """
#     Normalizes each trace in the profile trace array based on their center values, where the figure should agree.
#     """
#     center_values = prof_trace_array[:, int(prof_trace_array.shape[1]/2)]
#     mean_center_val = np.nanmean(center_values)
#     norm_prof_trace_array = np.copy(prof_trace_array)
#     for i in range(norm_prof_trace_array.shape[0]):
#         norm_prof_trace_array[i] += mean_center_val - center_values[i]
#     return norm_prof_trace_array

# def arrange_traces_on_image(prof_trace_array, trace_sep_angle=20.):
#     """
#     The first trace (which corresponds to 0 degrees) runs left to right horizontally through the center of the image. Sequential traces are placed counterclockwise
#     """
#     N_traces = prof_trace_array.shape[0]
#     N_points = prof_trace_array.shape[1]
#     prof_image = np.full((N_points, N_points), np.nan)
#     angles = np.arange(0, N_traces*trace_sep_angle, trace_sep_angle)
#     image_center = (N_points//2, N_points//2) # (x, y)
#     for i in range(N_traces):
#         profile = prof_trace_array[i]
#         angle = angles[i]
#         rad_angle = np.deg2rad(angle)
#         # print('='*20+'Trace {}'.format(i+1)+'='*20)
#         if angle % 180 == 0:
#             start_offset = -(N_points // 2)
#         else:
#             start_offset = -(N_points // 2 - 1)
#         for j, prof_value in enumerate(profile):
#             distance = start_offset + j
#             x = int(math.floor(image_center[0] + distance*np.cos(rad_angle)))
#             y = int(math.floor(image_center[1] - distance*np.sin(rad_angle)))
#             placement = False
#             # x = int(image_center[0] + distance*np.cos(rad_angle))
#             # y = int(image_center[1] - distance*np.sin(rad_angle))
#             if 0<=x<N_points and 0<=y<N_points:
#                 prof_image[y, x] = prof_value
#                 placement = True
#             # print('val: {:.1f}, (y, x): ({}, {}), placement: {}'.format(prof_value, y, x, placement))
#     return prof_image

# def interpolate_prof_image(prof_image, interp_method='linear'):
#     """
#     Interpolates between the traces of a 2D profilometer image
#     """
#     non_nan_mask = ~np.isnan(prof_image)
#     y_non_nan, x_non_nan = np.where(non_nan_mask)
#     values_non_nan = prof_image[non_nan_mask]
#     x_full, y_full = np.meshgrid(np.arange(prof_image.shape[0]), np.arange(prof_image.shape[1]))
#     interp_prof_image = scipy.interpolate.griddata((x_non_nan, y_non_nan), values_non_nan, (x_full, y_full), method=interp_method)
#     return interp_prof_image
