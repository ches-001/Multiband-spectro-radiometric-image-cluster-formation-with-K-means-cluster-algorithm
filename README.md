<h1>Multi-band Spectro Radiomertric Image Analysis with K-means Cluster Algorithm </h1>

<h2 style="color:blue;">Overview</h2>

<p>Multi-band Spectro Radiomertric images are images comprising of several channels / bands which hold information on band energy in each pixel.<br>
The most common multi band channels are the RGB (Red Green Blue) channels of the visible light spectrum.</p>

<p>The images used are LANDSAT 8 satellite images and each image consist of three bands, namely: Thermal Infrared, Red and Near infrared bands corresponding to band 10, band 4 and band 5 of LANDSAT 8 satellite imagery with wavelengths of 10.895µm, 0.655µm and 0.865µm respectively.</p>

<p>Each pixel in each bands of each image are used to compute three features namely: NDVI (Normalized Differential Vegetative Index), PV (Portion of Vegetation) and LST (Land Surface Temperature).</p>

<p>The K-means cluster algorithm is initialized and the "number of clusters" hyper-parameter is set to 60.
The algorithm then trains on the extracted features and forms 60 different clusters represented by each of the 60 centroids.</p>

<p>These centroids are stored in the "ouput" folder and <strong>will be futher studied to learn what NDVI, PV and LST combinations a geograhical location might need to have for the occurence and spread of wild fire to be highly probable</strong>.</p>
<br><br>

<h2 style="color:blue;">Features</h2>

<h4 style="color:yellow;">NDVI (Normalized Differential Vegetative Index):</h4>
<p>The Normalized Differential Vegetative Index is a metric for checking the presence and health of a vegetation in a given region.<br>
It is basically how much RED light energy from the visible light spectrum is absorbed by the plant and how much NIR (near-infrared rays) it emmits.<br>
Healthy vegetation absorbs red-light energy to fuel photosynthesis and create chlorophyll, and a plant with more chlorophyll will reflect more near-infrared energy than an unhealthy plant.<br> 
The NDVI ranges from -1 to 1, -1 corresponds to a very unhealthy plant and 1 corresponds to a very healthy plant.<br>
</p>
<p>
The mathematical expression for NDVI is:<br>
$$NDVI = (NIR - RED) / (NIR + RED)$$
</p><br>


<h4 style="color:yellow;">PV (Portion of Vegetation):</h4>
<p>
Portion of Vegetation is the ratio of the vertical projection area of vegetation on the ground to the total vegetation area
</p>
<p>
The mathematical expression for PV is:<br>
$$PV = (NDVI - NDVImin) / (NDVImin + NDVImax)$$<br>
NDVImin is the minimum NDVI value a pixel holds in a single image<br>
NDVImin is the maximum NDVI value a pixel holds in a single image
</p><br>


<h4 style="color:yellow;">LST (Land Surface Temperature):</h4>
<p>
Land Surface Temperature is the radiative temperature / intensity of the land surface
</p>
<p>
The mathematical expression for LST is:<br>
<strong>LST = BT / ( 1 + ( ( kn * BT / p ) * np.log(E) ) )</strong><br>

**BT** is brighness Temperature in celcius and is mathematically expressed as:<br>
**BT = (K2 / np.log( ( K1 / TOA ) + 1 )) - 273.15**<br>
where K1 and K2 are landsat 8 constants 774.8853 and 1321.0789 respectively<br><br>

**TOA** (Top of Atmosphere) Reflectance is a unitless measurement which provides the ratio of radiation reflected to the incident solar radiation on a given surface.<br>
It is mathematically expressed as:<br>
**TOA = ML * TIR + Al**<br>
where ML and Al are landsat 8 constants 3.42E-4 and 0.1 respectively.<br><br>

**p** is mathematically expressed as:<br>
**p = hc/A**<br>
where h, c and a are plank's constant, speed of light and boltzmann constant respectively<br><br>

**E** is emissivity of the land surface and is mathematically expressed as:<br>
**( Ev * PV * Rv ) + ( Es * ( 1 - PV ) * Rs ) + C**<br>
where:<br>
Ev (Vegitation Emissivity) of location = 0.986<br>
Es (Soil Emissivity) of location = 0.973<br>
C (topography factor) of location = 0.0001<br>
Rv =(0.92762 + (0.07033*PV))<br>
Rs=(0.99782 + (0.05362*PV))<br>
</p>
<br><br>

<h2 style="color:blue;">Dependencies</h2>
<ul>
<li>Rasterio</li>
<li>Numpy</li>
<li>Pandas</li>
<li>Sklearn</li>
<li>Pickle</li>
</ul>
<br><br>

<h2 style="color:blue;">Setup</h2>
clone the repository and download the 'requirement.txt' files, then open terminal in the working directory and  type <strong>'pip install -r requirements.txt'<strong> to install all the requirements for this project.
