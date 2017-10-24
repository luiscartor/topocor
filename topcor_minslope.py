## ====================================================================================
## topcor_minslope.py
## by Luis
## 23 October 2017
##
## Script for topographic correction using minnaert algorithm with k coefficient
## dependent on the slope range
## ====================================================================================

try: #indent rest of the code when uncommented

    import arcpy, os, random
    import numpy as np
    from arcpy.sa import *
    arcpy.CheckOutExtension('Spatial')
    from math import*
    deg2rad = math.pi / 180.0

    ## 1. INPUTS

    ## 1.1 Sets working directory
    # Asks for working directory
    #arcpy.env.workspace = raw_input('Enter a working folder (e.g. C:\\LCM2015\\data\\205_11\\): ')

    # Maunal input of working directory
    arcpy.env.workspace = 'C:\\topcorscripts'
    arcpy.env.overwriteOutput = True

    ## 1.2 Data inputs
    # Asks for satelite image path
    #inImage = raw_input('Enter a satellite image (e.g. 205_011_cm.tif or C:\\LCM2015\\data\\205_011\\205_011_cm.tif if the image is in different path from working directory): ')
    inImage = 'C:\\topcorscripts\\30UWE20Apr16_sen2cor_10m.tif'

    # Asks for DEM path
    #inDem = raw_input('Enter a DEM (e.g. GBdtm.tif or C:\\LCM2015\\DTM\\GBdtm.tif if the DTM is in different path from working directory): ')
    inDem = 'C:\\topcorscripts\\GBdtm3.tif'

    # Asks for sun elevation
    #inElev = raw_input('Enter sun elevation in degrees (elevation = 90 - zenith angle): ')
    inElev = 43.56

    # Asks for sun azimuth
    #inAz = raw_input('Enter sun azimuth in degrees: ')
    inAz = 165.25

    # Number of slope ranges
    rngs = 8

    # Dividing factor for representing reflectance
    ref_fac = 300.0


    ## 2. READ AND PREPARE DATA
    ## 2.1 Read data
    satImage = arcpy.Raster(inImage)
    dem = arcpy.Raster(inDem)
    zenith = radians(90.0 - float(inElev))
    az = radians(float(inAz))

    ## 2.2 Slope and aspect preparation
    # Matches the dem with the satellite image
    # DEM reprojection and resampling
    projection = arcpy.Describe(satImage).spatialReference
    resolution = satImage.meanCellWidth
    arcpy.ProjectRaster_management(dem, 'dem_proj.tif', projection, 'BILINEAR', resolution)
    arcpy.env.outputCoordinateSystem = projection

    # DEM crop to satellite image extent
    dem_proj = arcpy.Raster("dem_proj.tif")
    dem = ExtractByMask(dem_proj, satImage)
    del dem_proj

    # Creates slope and aspect rasters and convert to arrays
    slope = Slope(dem)
    aspect = Aspect(dem)
    del dem

    ## 2.3 Ilummination variable preparation
    # Calculation of the illumination(Holben and Justice, 1980)
    # cos(i) = cos(sun.zenith)cos(slope) + sin(sun.zenith)sin(slope)cos(sun.az-aspect)
    # cos(i) varies from -1 to 1
    cos_il = (Cos(zenith)* Cos(deg2rad*slope)) + (Sin(zenith)*Sin(deg2rad*slope)*Cos(az-(deg2rad*aspect)))
    del aspect
    # Avoid zero values (avoids dividing by zero)
    cos_il = Con(cos_il == 0, 0.001, cos_il)
    # Avoid extreme dark values
    cos_il = Con((cos_il < 0.3) & (cos_il > 0), 0.3, cos_il)
    cos_il = Con((cos_il > -0.3) & (cos_il < 0), -0.3, cos_il)


    ## 3. MAIN ROUTINE
    ## Corrects topography effect using Minnaert correction with slope algorithm(based on Lu et al. 2008) 

    # Loops throughout each band
    desc = arcpy.Describe(satImage)
    result_rasters = [None] * desc.bandCount

    for idx,band in enumerate(desc.children):
        
        print ("Processing band"), idx+1
        bandName = band.name
        sat_band = Raster (os.path.join(inImage,band.name))

        ## 3.1 Sample data to calculate k coefficients
        # Extracts sample data
        sample_size = 50000
        # Because we might have many na values, we sample 30 times more pixels than needed samples
        sample_pixels = sample_size * 30

        # Sample arrays for reflectance values, slope and cos(ilumination)
        band_ar = []
        slope_ar = []
        cosil_ar = []

        ## 3.1.1 Loops over band image blocks to sample pixels, so the arrays are smaller (avoid memory error)
        # Also we sample over the whole image
        blocksize = 1000
        nblocks = len(range(0, sat_band.height, blocksize))* len(range(0, sat_band.width, blocksize))

        for x in range(0, sat_band.width, blocksize):
            for y in range(0, sat_band.height, blocksize):

                # Lower left coordinate of block (in map units)
                mx = sat_band.extent.XMin + x * sat_band.meanCellWidth
                my = sat_band.extent.YMin + y * sat_band.meanCellHeight
                # Upper right coordinate of block (in cells)
                lx = min([x + blocksize, sat_band.width])
                ly = min([y + blocksize, sat_band.height])
                # noting that (x, y) is the lower left coordinate (in cells)
                band_blockar = arcpy.RasterToNumPyArray(sat_band, arcpy.Point(mx, my), lx-x, ly-y, nodata_to_value=0)
                slope_blockar = arcpy.RasterToNumPyArray(slope, arcpy.Point(mx, my), lx-x, ly-y, nodata_to_value=0)
                cosil_blockar = arcpy.RasterToNumPyArray(cos_il, arcpy.Point(mx, my), lx-x, ly-y, nodata_to_value=0)
                #  Pixels per block being sampled
                rancells = np.random.randint(0,band_blockar.size,sample_pixels/nblocks)

                # We append the sampled pixel values to arrays
                band_ar = np.append(band_ar, np.take(band_blockar,rancells))
                slope_ar = np.append(slope_ar, np.take(slope_blockar,rancells))
                cosil_ar = np.append(cosil_ar, np.take(cosil_blockar,rancells))
                

        ## 3.1.2 Compiles sampled pixels from all blocks
        # Exctracts the values for slope, cos_il and sat image from all blocks
        slope_sam = np.empty(sample_size)
        cosil_sam = np.empty(sample_size)
        band_sam = np.empty(sample_size)
        j = 0
        # Creates sample vector from random cells values
        for i in np.random.permutation(slope_ar.size):
            if (slope_ar.item(i)>0 and cosil_ar.item(i)>0 and band_ar.item(i)>0):
                slope_sam[j] = slope_ar.item(i)
                cosil_sam[j] = cosil_ar.item(i)
                band_sam[j] = band_ar.item(i)
                j += 1
            if j == sample_size:
                break

        ## 3.1.3 The data is divided in 8 slope ranges and the k calculation is performed
        ranges = rngs
        # We find the maximum slope to split the slope classes
        max_slope = np.amax(slope_sam)

        ## 3.1.3.1 For each range we calculate the k coefficient
        k = np.empty(ranges)
        meansl = np.empty(ranges)
        kranges = np.empty(ranges,dtype=[('x', float), ('y', float), ('z', int)])

        # Loops for every slope range and calculate k
        for i in range(ranges):
            
            sl = slope_sam[np.where(np.logical_and(slope_sam>=(max_slope/ranges)*i,slope_sam<(max_slope/ranges)+(max_slope/ranges)*i))]
            ci = cosil_sam[np.where(np.logical_and(slope_sam>=(max_slope/ranges)*i,slope_sam<(max_slope/ranges)+(max_slope/ranges)*i))]
            re = band_sam[np.where(np.logical_and(slope_sam>=(max_slope/ranges)*i,slope_sam<(max_slope/ranges)+(max_slope/ranges)*i))]   

            # If there are less than 20 pixels for the slope range,
            # a k coefficient of the previous range is assigned (for the first range, k is always calculated)
            if (i == 0 or sl.size>20):
                ## Calculates k for the slope range. From Lu et al. 2008 Equation 3.
                # Fits a linear regression of the logarithms to find the slope
                k_rg = np.polyfit(np.log10(np.cos(deg2rad*sl)*ci),np.log10((re/100.0)*np.cos(deg2rad*sl)),1)[0]

                # Array with k values: k coefficient ranges between 0 and 1
                if (k_rg > 1):
                    k[i] = 1
                elif (k_rg < 0):
                    k[i] = 0
                else:
                    k[i] = k_rg
                
            #If there are less than 20 pixels for that class, the k of the previous class is assigned
            else:
                k[i] = k[i-1]

            # Stores the k values for every range in an array 
            kranges[i] = ((max_slope/ranges)*i,(max_slope/ranges)+(max_slope/ranges)*i,k[i]*10000.0)
            

        ## 3.2 A raster map of k values is created
        # Reclassifies slope values into k values
        remap = RemapValue(kranges)
        k_raster = Reclassify(slope,'Value',remap,'NODATA')

        ## 3.3 The minnaert correction with slope is applied to the target band

        # Minnaert with slope equation
        # Reflectance is divided by 300 to represent reflectance values for Sentinel 2 images
        cor_refl = (sat_band/ref_fac) * ((Cos(zenith)/cos_il)**((k_raster/10000.0)))

        # The corrected band is stored as a 16 Bit raster in a list
        result_rasters[idx] = Int(cor_refl*ref_fac + 0.5)
        

    ## 4. Stack all bands and save
    name = os.path.join(os.path.splitext(inImage)[0] + "_tc.tif")
    arcpy.CompositeBands_management(result_rasters, name)


except:
    print "Processing failed"
    print arcpy.GetMessages()

    
