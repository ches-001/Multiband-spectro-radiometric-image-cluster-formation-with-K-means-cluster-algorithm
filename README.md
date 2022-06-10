# Multi-band Spectro Radiomertric Image Analysis with K-means Cluster Algorithm

Multi-band Spectro Radiomertric images are images comprising of several channels / bands which hold information on band energy in each pixel.
The most common multi band channels are the RGB (Red Green Blue) channels of the visible light spectrum.

The images used are LANDSAT 8 satellite images and each image consist of three bands, namely: Thermal Infrared, Red and Near infrared bands corresponding to band 10, band 4 and band 5 of LANDSAT 8 satellite imagery with wavelengths of 10.895µm, 0.655µm and 0.865µm respectively.

Each pixel in each bands of each image are used to compute three features namely: NDVI (Normalized Differential Vegetative Index), PV (Portion of Vegetation) and LST (Land Surface Temperature).

The K-means cluster algorithm is initialized and the "number of clusters" hyper-parameter is set to 60.
The algorithm then trains on the extracted features and forms 60 different clusters represented by each of the 60 centroids.

These centroids are stored in the "ouput" folder and will be futher studied to learn what NDVI, PV and LST combinations a geograhical location might need to have for the occurence and spread of wild fire to be highly probable.


## Features

### NDVI (Normalized Differential Vegetative Index):
The Normalized Differential Vegetative Index is a metric for checking the presence and health of a vegetation in a given region.<br>
It is basically how much RED light energy from the visible light spectrum is absorbed by the plant and how much NIR (near-infrared rays) it emmits.<br>
Healthy vegetation absorbs red-light energy to fuel photosynthesis and create chlorophyll, and a plant with more chlorophyll will reflect more near-infrared energy than an unhealthy plant.
The NDVI ranges from -1 to 1, -1 corresponds to a very unhealthy plant and 1 corresponds to a very healthy plant.

The mathematical expression for NDVI is:
$$ NDVI = (NIR - RED) / (NIR + RED) $$



### PV (Portion of Vegetation):
Portion of Vegetation is the ratio of the vertical projection area of vegetation on the ground to the total vegetation area
The mathematical expression for PV is:
$$ PV = (NDVI - NDVImin) / (NDVImin + NDVImax) $$
NDVImin is the minimum NDVI value a pixel holds in a single image
NDVImin is the maximum NDVI value a pixel holds in a single image



### LST (Land Surface Temperature):
Land Surface Temperature is the radiative temperature / intensity of the land surface

The mathematical expression for LST is:
$$ LST = BT / ( 1 + ( ( kn * BT / p ) * np.log(E) ) ) $$
**BT** is brighness Temperature in celcius and is mathematically expressed as:
$$ BT = (K2 / np.log( ( K1 / TOA ) + 1 )) - 273.15 $$
where K1 and K2 are landsat 8 constants 774.8853 and 1321.0789 respectively

**TOA** (Top of Atmosphere) Reflectance is a unitless measurement which provides the ratio of radiation reflected to the incident solar radiation on a given surface.

It is mathematically expressed as:
$$ TOA = ML * TIR + Al $$
where ML and Al are landsat 8 constants 3.42E-4 and 0.1 respectively.

**p** is mathematically expressed as:
$$ p = hc/A $$
where h, c and a are plank's constant, speed of light and boltzmann constant respectively

**E** is emissivity of the land surface and is mathematically expressed as:<br>
$$ ( Ev * PV * Rv ) + ( Es * ( 1 - PV ) * Rs ) + C $$
where:
$$ Ev (Vegitation Emissivity) of location = 0.986 $$
$$ Es (Soil Emissivity) of location = 0.973 $$
$$ C (topography factor) of location = 0.0001 $$
$$ Rv = (0.92762 + (0.07033*PV)) $$
$$ Rs = (0.99782 + (0.05362*PV)) $$



## Dependencies
<ul>
<li>Rasterio</li>
<li>Numpy</li>
<li>Pandas</li>
<li>Sklearn</li>
<li>Pickle</li>
</ul>


## Setup
clone the repository and download the 'requirement.txt' files, then open terminal in the working directory and  type <strong>'pip install -r requirements.txt'<strong> to install all the requirements for this project.
