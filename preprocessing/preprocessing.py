import snappy
from snappy import ProductIO
from snappy import HashMap
from snappy import GPF
from snappy import ProgressMonitor
from snappy import WriteOp
from snappy import File
from snappy import System
from snappy import PrintStream
from snappy import NullStream
from snappy import Logger
from snappy import Level
import math
import numpy as np
import time
import os
import gc
from lmfit import Model
from numpy import exp, pi, sqrt
from netCDF4 import Dataset
import random
import csv


# ----------------------------------------------------------------
# ------------------------- OPTIONS ------------------------------
# ----------------------------------------------------------------


# If images are to be written, for CNN
will_write_img = True


# If CSV file is to be written, for regression network
will_write_csv = False


# If azimuth cutoff should be included
include_azm = False


# If the dataset should be evenly distributed across all wave heights
even_distribution = False


# If the selection process should be randomized for common wave heights, only works with even distribution
slow_down_selection = False


# Amount of pixels in each sub-image
n = 200


# Size of each bucket of waves in meters
bucket_size = 0.2


# Maximum amount of waves per bucket
bucket_max = 8000


# Minimum and maximum wave height to be sorted in meters
min_wave = 0
max_wave = 4


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------


# Number of buckets in array
bucket_arr_size = int((max_wave - min_wave)/bucket_size)


# Suppresses a bunch of messages
Logger.getLogger('').setLevel(Level.OFF)
snappy.SystemUtils.LOG.setLevel(Level.OFF)
System.setOut(PrintStream(NullStream()))
System.setErr(PrintStream(NullStream()))


# Paths to data folders
unprocessed_path = r'data\s1_unprocessed'
preprocessed_path = r'data\s1_preprocessed'
output_path = r'data\img_output'
shapefile = r'data\shapefile\GSHHS_f_L1.shp'
output_csv = r'data\csv_output\params.csv'
cop_file = r'data\model_data\majsep2020.nc'
numpy_path = r'memory\amt_in_buckets.npy'
processed_files_path = r'memory\processed_files.npy'
processed_rows_path = r'memory\processed_rows.npy'


# Creates a dataset for the wave model
copernicus = Dataset(cop_file, mode='r')
time_len = len(copernicus.variables['time'][:])
time_arr = np.zeros(time_len)


# Checks what year we are using, for creating time array
if cop_file == r'data\model_data\majsep2020.nc':
    time_arr[0] = 1588284000
elif cop_file == r'data\model_data\majsep2021.nc':
    time_arr[0] = 1619820000


# Creates a time array for matching
for i in range(time_len - 1):
    time_arr[i + 1] = time_arr[i] + 3600


# Reads variables from Copernicus file
lons = copernicus.variables['lon'][:]
lats = copernicus.variables['lat'][:]
VHM0 = copernicus.variables['VHM0'][:]
Time = copernicus.variables['time'][:]


# Loads numpy vector with information about number of images per wave height bucket
if not os.path.exists(numpy_path):
    empty_arr = np.zeros(bucket_arr_size + 1, dtype=int)
    np.save(numpy_path, empty_arr)


# Creates numpy file with information about processed files
if not os.path.exists(processed_files_path):
    empty_arr = np.array([], dtype='object')
    np.save(processed_files_path, empty_arr)


# Creates numpy file with information about processed rows in file
if not os.path.exists(processed_rows_path):
    empty_arr = np.zeros(1)
    np.save(processed_rows_path, empty_arr)


# Help variables for printing
param_val = 0
times = np.empty(240)
times[:] = np.NAN
index = 0


# ----------------------------------------------------------------
# --------------------------HELPERS-------------------------------
# ----------------------------------------------------------------


# Returns the value of an upside down bell curve
def probability_curve(wave_height):
    return min(-math.exp(-(((wave_height - 0.682)**10)/0.0001)) + 1.1, 1)


# Determines if a wave is rejected or not given its probability
def decision(wave_height):
    return random.random() < probability_curve(wave_height)


# Calculates the azimuth_cutoff
def azimuth_cutoff(band, x, y):
    xdata = np.arange(-5 * n, 5 * n, 10)
    acf_array = np.zeros((n, n), np.float32)
    band.readPixels(x, y, n, n, acf_array)
    spectral_density = np.square(np.abs(np.fft.fft2(acf_array)))
    spec_dens_x = np.mean(spectral_density, axis=0)
    AACF = np.fft.ifft(spec_dens_x)
    AACF = np.fft.fftshift(AACF)
    AACF = np.abs(AACF)
    AACF = (AACF - min(AACF))  # /np.max(AACF)
    gmodel = Model(gaussian)
    result = gmodel.fit(AACF, x=xdata, amp=1, cen=0, wid=10)
    acw = np.sqrt(2) * np.pi * np.asarray(result.params)[2]
    return acw


# Creates a gaussian fit for azimuth cutoff
def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2 * pi) * wid)) * exp(-(x - cen) ** 2 / (2 * wid ** 2))


# Prints information about processed files
def print_info_of_row(count, row, dim_y, kept_land, kept_homo, thrown_land, thrown_homo, filled_buckets, processed):

    if row == 0:
        print("File: " + str(count), "Percentage of file: " + str(truncate(float(row * 100) / dim_y, 1)) + "%")
    else:

        global index

        print("File: " + str(count), "File progress: " + str(truncate(float(row * 100) / dim_y, 1)) + "%",
              "Land: " + str(truncate(float(kept_land) * 100 / (kept_land + thrown_land), 1)) + "%",
              "Homogeneous: " + str(truncate(float(kept_homo) * 100 / (kept_homo + thrown_homo), 1)) + "%",
              "Buckets filled: " + str(filled_buckets) + '/' + str(bucket_arr_size),
              "Processed imgs: " + str(processed))


# Returns a SNAP product from subregion of source product
def create_subset(source, x, y, x_offset, y_offset, copy_boolean):
    parameters = HashMap()
    parameters.put('copyMetadata', copy_boolean)
    parameters.put('region', '%s, %s, %s, %s' % (x, y, x_offset, y_offset))

    return GPF.createProduct("Subset", parameters, source)


# Homogeneity test, determines if an image has to many disturbances, such as boats
def is_homogeneous(band, x, y):

    # Will subdivide image into 5x5 grid
    squares_per_side = 5

    # The two sums used when calculating the homogeneity parameter
    mean_sum, mean_var_sum = 0, 0

    matrix_vec = []

    # Pixels per side
    length = int(n/squares_per_side)

    for i in range(squares_per_side):
        for j in range(squares_per_side):

            band_data = np.zeros((length, length), np.float32)

            # Reads the pixels for each sub-image
            band.readPixels(x + i*length, y + j*length, length, length, band_data)

            # Calculate spectral density of sub-image
            spec_dense = np.square(np.abs((np.fft.ifft2(band_data))))
            matrix_vec.append(spec_dense)

    for i in range(squares_per_side):
        for j in range(squares_per_side):

            wave_number_vec = np.zeros(squares_per_side ** 2)

            for k, matrix in enumerate(matrix_vec):
                wave_number_vec[k] = matrix[i][j]

            mean = np.mean(wave_number_vec)
            var = np.var(wave_number_vec)

            # Adds the sub-image's data to the running sums
            mean_sum += mean
            mean_var_sum += var/mean

    # Calculates the final homogeneity parameter for this imuage
    param = (mean_var_sum / mean_sum)

    return param < 0.2


# help function to truncate decimals
def truncate(number, digits):
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


# Snappy function to import vectors
def import_vector(source):
    parameters = HashMap()
    parameters.put('vectorFile', shapefile)
    parameters.put('separateShapes', False)
    output = GPF.createProduct('Import-Vector', parameters, source)
    return output


# Creates a landmask from a given product
def create_land_mask(source):
    parameters = HashMap()
    parameters.put('landMask', False)
    parameters.put('useSRTM', False)
    parameters.put('geometry', 'GSHHS_f_L1')
    parameters.put('invertGeometry', True)
    parameters.put('shorelineExtension', 0)
    parameters.put('useVectorasMask', False)

    masked = GPF.createProduct('Land-Sea-Mask', parameters, source)
    return masked


# Checks if land exists anywhere in a box from coordinates
def land_exists(band, x_start, y_start):
    band_data = np.zeros((n, n), np.float32)
    band.readPixels(x_start, y_start, n, n, band_data)

    return np.min(band_data) == 0.0


# ----------------------------------------------------------------
# --------------------------PREPROCESSING-------------------------
# ----------------------------------------------------------------


# Applies orbit file to source
def do_apply_orbit_file(source):
    print('\tApply orbit file...')
    parameters = HashMap()
    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

    parameters.put('orbitType', 'Sentinel Restituted (Auto Download)')
    parameters.put('polyDegree', '3')
    parameters.put('continueOnFail', 'true')

    output = GPF.createProduct('Apply-Orbit-File', parameters, source)
    return output


# Does thermal noise removal on source
def do_thermal_noise_removal(source):
    print('\tThermal noise removal...')
    parameters = HashMap()
    parameters.put('removeThermalNoise', True)
    output = GPF.createProduct('ThermalNoiseRemoval', parameters, source)
    return output


# Does radiometric calibration
def do_calibration(source, polarization, pols):
    print('\tCalibration...')
    parameters = HashMap()
    parameters.put('outputSigmaBand', True)

    if polarization == 'DH':
        parameters.put('sourceBands', 'Intensity_HH,Intensity_HV')
    elif polarization == 'DV':
        parameters.put('sourceBands', 'Intensity_VH,Intensity_VV')
    elif polarization == 'SH' or polarization == 'HH':
        parameters.put('sourceBands', 'Intensity_HH')
    elif polarization == 'SV':
        parameters.put('sourceBands', 'Intensity_VV')
    else:
        print("different polarization!")

    parameters.put('selectedPolarisations', pols)
    parameters.put('outputImageScaleInDb', False)
    output = GPF.createProduct("Calibration", parameters, source)
    return output


# Does speckle filtering
def do_speckle_filtering(source):
    print('\tSpeckle filtering...')
    parameters = HashMap()
    parameters.put('filter', 'Lee')
    parameters.put('filterSizeX', 5)
    parameters.put('filterSizeY', 5)
    output = GPF.createProduct('Speckle-Filter', parameters, source)
    return output


# ----------------------------------------------------------------
# --------------------------ALGORITHM-----------------------------
# ----------------------------------------------------------------

def main():

    # Loads file containing information about processed files
    processed_files = np.load(processed_files_path, allow_pickle=True)
    count = processed_files.size + 1

    # Main loop through all unprocessed SAR images
    for folder in os.listdir(unprocessed_path):
        if folder in processed_files:
            continue

        print("-----------------------------------------------------------")
        print('Working on: ' + folder)

        # Number of pictures kept after homogeneous test
        kept_land = 1
        kept_homo = 1

        # Number of pictures thrown after homogeneous test
        thrown_land = 1
        thrown_homo = 1

        # ----------------------------------------------------------------
        # ------------------------CREATING SUBSETS------------------------
        # ----------------------------------------------------------------

        # Garbage collection
        gc.enable()
        gc.collect()

        # Reads the SAR image
        sentinel_1 = ProductIO.readProduct(unprocessed_path + "\\" + folder + "\\manifest.safe")

        # Extracts the polarization as a string
        polarization = folder.split("_")[3][2:4]

        if polarization == 'DV':
            pols = 'VH,VV'
        elif polarization == 'DH':
            pols = 'HH,HV'
        elif polarization == 'SH' or polarization == 'HH':
            pols = 'HH'
        elif polarization == 'SV':
            pols = 'VV'
        else:
            print("Polarization error!")

        # Starts a chain of preprocessing, each step takes the previous product as input
        orbital = do_apply_orbit_file(sentinel_1)
        thermal = do_thermal_noise_removal(orbital)
        s1_preprocessed = do_calibration(thermal, polarization, pols)

        del orbital
        del thermal

        # Removes the band that we're not using
        s1_preprocessed.removeBand(s1_preprocessed.getBand("Sigma0_VH"))

        # Gets the number of pixels in the image, x then y
        x_amt = int(s1_preprocessed.getSceneRasterWidth())
        y_amt = int(s1_preprocessed.getSceneRasterHeight())

        # Calculates number of whole squares that can fit in the picture, each one being nxn pixels
        dim_x = int(math.floor(x_amt / n))
        dim_y = int(math.floor(y_amt / n))

        # Copy of product where the land is filtered out and set as 0.0
        mask = create_land_mask(import_vector(s1_preprocessed))

        # Holds the actual data from the land mask
        band = mask.getBand("Sigma0_VV")

        print("-----------------------------------------------------------")
        print("Preprocessing done!")
        print("Will subdivide image into %s sub-images with dimensions %s x %s" % (dim_x * dim_y, dim_x, dim_y))
        print("-----------------------------------------------------------")

        # Measures time per file
        start_time_file = time.time()

        # Main loop through every individual SAR image
        for row in range(dim_y):

            # Prints information about current file
            amt_in_buckets = np.load(numpy_path)
            processed_row = np.load(processed_rows_path)[0]

            if row < processed_row:
                continue

            print_info_of_row(count, row, dim_y, kept_land, kept_homo, thrown_land, thrown_homo, np.count_nonzero(amt_in_buckets[0:bucket_arr_size-1] == bucket_max), amt_in_buckets[bucket_arr_size])

            for col in range(dim_x):

                # Upper left corner of current sub-image
                x, y = col * n, row * n

                # Checks for every square in the image if it contains land and is homogeneous
                if not land_exists(band, x, y):

                    thrown_land += 1

                    if is_homogeneous(band, x, y):
                        kept_homo += 1

                        # Creates the subset
                        subset = create_subset(s1_preprocessed, x, y, n, n, False)

                        # Checks for reasonable azimuth value
                        if include_azm:
                            azimuth_value = azimuth_cutoff(band, x, y)
                            if not 50 <= azimuth_value <= 250:
                                continue

                        # Converts metadata time to unix time
                        date_as_unix = int(subset.getMetadataRoot().getElementAt(1).getAttributeUTC("first_line_time").getAsDate().getTime() / 1000)
                        lat_tl = subset.getMetadataRoot().getElementAt(1).getAttributeDouble("first_near_lat")
                        lat_tr = subset.getMetadataRoot().getElementAt(1).getAttributeDouble("first_far_lat")
                        lat_br = subset.getMetadataRoot().getElementAt(1).getAttributeDouble("last_far_lat")
                        lat_bl = subset.getMetadataRoot().getElementAt(1).getAttributeDouble("last_near_lat")
                        long_tl = subset.getMetadataRoot().getElementAt(1).getAttributeDouble("first_near_long")
                        long_tr = subset.getMetadataRoot().getElementAt(1).getAttributeDouble("first_far_long")
                        long_br = subset.getMetadataRoot().getElementAt(1).getAttributeDouble("last_far_long")
                        long_bl = subset.getMetadataRoot().getElementAt(1).getAttributeDouble("last_near_long")

                        # Number of decimals in coordinates
                        nr_dec = 4

                        # Saves coordinate as strings
                        min_lat = truncate(min(lat_tl, lat_tr, lat_bl, lat_br), nr_dec)
                        max_lat = truncate(max(lat_tl, lat_tr, lat_bl, lat_br), nr_dec)
                        min_long = truncate(min(long_tl, long_tr, long_bl, long_br), nr_dec)
                        max_long = truncate(max(long_tl, long_tr, long_bl, long_br), nr_dec)

                        # Chooses the closest time available from the model_data data
                        absolute_difference_function = lambda list_value: abs(list_value - float(date_as_unix))
                        closest_time = min(time_arr, key=absolute_difference_function)
                        time_index = int(np.where(time_arr == closest_time)[0])

                        # Sets index for lats and longs
                        lat_low_ind = np.argmin(np.abs(lats - min_lat))
                        lat_upp_ind = np.argmin(np.abs(lats - max_lat))
                        long_low_ind = np.argmin(np.abs(lons - min_long))
                        long_upp_ind = np.argmin(np.abs(lons - max_long))

                        # Approximated wave height
                        wave_approx = np.mean(copernicus.variables['VHM0'][time_index, lat_low_ind:lat_upp_ind, long_low_ind:long_upp_ind])

                        # Errors occurs sometimes in matching, this discards those
                        if type(wave_approx) == np.ma.core.MaskedConstant:
                            continue

                        # Discards images that does not fit in a bucket or is randomized away
                        if even_distribution:

                            # Calculates which bucket the wave height will be sorted into
                            bucket_index = 0

                            if wave_approx > max_wave:
                                continue

                            counter = 0

                            for i in range(min_wave*10, max_wave*10, int(bucket_size*10)):
                                if i/10 <= wave_approx <= i/10 + bucket_size:
                                    bucket_index = counter
                                    continue
                                counter += 1

                            # Loads the bucket data and checks if that bucket is full
                            amt_in_buckets = np.load(numpy_path)

                            if amt_in_buckets[bucket_index] >= bucket_max:
                                continue

                            if slow_down_selection:
                                if not decision(wave_approx):
                                    continue

                            amt_in_buckets[bucket_index] = amt_in_buckets[bucket_index] + 1
                            amt_in_buckets[bucket_arr_size] = amt_in_buckets[bucket_arr_size] + 1
                            np.save(numpy_path, amt_in_buckets)

                        # Loads the data
                        subset_band = subset.getBand("Sigma0_VV")
                        subset_data = np.zeros((n, n), np.float32)
                        subset_band.readPixels(0, 0, n, n, subset_data)

                        if include_azm:
                            azm = str(truncate(azimuth_value, nr_dec))

                        # Calculates mean and variance of subimage
                        mean = str(truncate(np.mean(subset_data), nr_dec))
                        var = str(truncate(np.var(subset_data) * 1000, nr_dec))
                        wave_str = str(truncate(wave_approx, 8))

                        # Gives error message and discards image if time difference is to big
                        if abs(closest_time - date_as_unix) > 1800:
                            print('Time diff greater than 1800. Time diff: ' + str(abs(closest_time - date_as_unix)))
                            continue

                        # Creates the final subset without metadata
                        output_name = output_path + '\\' + wave_str + '_' + azm if include_azm else output_path + '\\' + wave_str

                        if will_write_img:
                            # Writes the final subset
                            write_op = WriteOp(subset, File(output_name), 'GeoTIFF')
                            write_op.writeProduct(ProgressMonitor.NULL)
                            write_op.dispose()
                            subset.dispose()


                        if will_write_csv:
                            with open(r'data\csv_output\params.csv', 'a', encoding='UTF8', newline='') as f:
                                writer = csv.writer(f)

                                if include_azm:
                                    writer.writerow([wave_approx, mean, var, azm])
                                else:
                                    writer.writerow([wave_approx, mean, var])
                    else:
                        thrown_homo += 1
                else:
                    kept_land += 1

            # Saves the row as processed
            np.save(processed_rows_path, np.array([row]))

        # Clears memory
        np.save(processed_rows_path, np.array([0]))
        sentinel_1.dispose()
        s1_preprocessed.dispose()
        mask.dispose()

        print("File done!", "Time elapsed: " + str(time.time() - start_time_file))
        count += 1

        # Saves the file as processed
        processed_files = np.load(processed_files_path, allow_pickle=True)
        processed_files = np.append(processed_files, folder)
        np.save(processed_files_path, processed_files)


if __name__ == "__main__":
    main()