#______________________________________________________________________________________________________________
#   Author:   ches-001                                                                                         |
#                                                                                                              |
#   Email:     ches.darksahades@gmail.com                                                                      |
#                                                                                                              |
#   contact@ : +2349057900367                                                                                  |
#                                                                                                              |
#   Project_type: Machine Learning                                                                             |
#______________________________________________________________________________________________________________|


#Dependencies:
import os
import rasterio
import numpy as np
import pandas as pd 
from sklearn.cluster import MiniBatchKMeans
import pickle
import time


dir_path = "data/imagery"
data = list()
for datum_path in os.listdir(dir_path):
    with rasterio.open(os.path.join(dir_path, datum_path), mode="r") as datum:
        data.append(datum.read())

TIR_BAND = np.squeeze(data[0]) #Thermal Infrared band (band 10: central wavelength = 10.895um)
RED_BAND = np.squeeze(data[1]) #Visible Light(RED) band (band 4: central wavelength = 0.655um)
NIR_BAND = np.squeeze(data[2]) #Near Infrared band (band 5: central wavelength = 0.865um)
n_features = 3



def data_preprocess(*array):
    r"""
    data_preprocess(*array) returns a preprocessed data by removing no-data values.

    parameter:
    -------------
    *array: ndarray
            dataset to be processed
    
    returns
    -------------
    processed array(s): ndarray
    """
    
    array = np.array(array, dtype=np.float32)
    array[array == 0] = np.nan
    return array


TIR_BAND, RED_BAND, NIR_BAND = data_preprocess(TIR_BAND, RED_BAND, NIR_BAND)



def computeNDVI(RED=RED_BAND, NIR=NIR_BAND):
    r"""
    computeNDVI(RED=RED_BAND, NIR=NIR_BAND) computes and returns the value of the NDVI
    (Normalized Differential Vegetative Index) Who's value ranges from (-1, 1).
    These values are computed with the following mathematical expression: NDVI = (NIR-RED)/(NIR+RED) where:
    NIR: is the amount of Near Infrared wave absorbed and dispersed by the vegetation,
    RED: is the amount of RED from the visible Light spectrum absorbed and reflected by the vegetation.

    parameters
    -------------
    RED: ndarray
        RED band (default = RED_BAND)
    NIR: ndarray
        NIR band (default = NIR_BAND)

    returns
    -------------
    NDVI: ndarray
        Normalized Differential Vegetative Index: (NIR-RED)/(NIR+RED)
    """
    return (NIR-RED)/(NIR+RED)



def computeTOA(Qcal=TIR_BAND, ML=3.42e-4, Al=0.1):
    r"""
    computeTOA() computes and returns the Top of Atmospheric Brightness
    with the mathematical expression: TOA (L) = Ml * Qcal + Al. where:
    ML: is Band-specific multiplicative rescaling factor from the metadata(RADIANCE_MULT_BAND_x,
    where x is the band number)
    Qcal: the band 10 imagery
    Al: is the Band-specific additive rescaling factor from the metadata (RADIANCE_ADD_BAND_x,
    where x is the band number).

    parameters
    -------------
    Qcal: ndarray
        TIR band (default = tir)
    ML: float
        Multiplicative Scale Factor from metadata (default = 0.000342)
    Al: float
        Additive Scale Factor from metadata (default = 0.1)

    returns
    -------------
    TOA = ndarray
        Top of Atmospheric Brightness: P ML * Qcal + Al
    """
    return ML * Qcal + Al



def computeBT(TOA=computeTOA(), K1=774.8853,  K2=1321.0789, kelvin_const=273.15):
    r"""
    computeBT(TOA=computeTOA(), K1=774.89,  K2=1321.08, kelvin_const=273.15)
    computes and returns the value of the BT (Brightness Temperature) ofthe reflectants
    using the mathematical expression BT = (K2 / (ln (K1 / L) + 1)) − 273.15, where:
    K1: is Band-specific thermal conversion constant from the metadata (K1_CONSTANT_BAND_x, 
    where x is the thermal band number).
    K2: is Band-specific thermal conversion constant from the metadata (K2_CONSTANT_BAND_x, 
    where x is the thermal band number).

    parameters
    -------------
    TOA: ndarray
          computed TOA (default = computeTOA())
    K1: float
        Thermal constant from metadata (default = 774.89)
    K2: float
        Thermal constant from metadata (default = 1321.08)
    kelvin_const: flaot
        physical constant (default = 273.15)

    returns
    -------------
    BT: ndarray
        Brightness Temperature: (K2 / np.log( ( K1 / TOA ) + 1 )) - kelvin_const
    """
    return (K2 / np.log( ( K1 / TOA ) + 1 )) - kelvin_const



def computePV(NDVI=computeNDVI()):
    r"""
    computePV(NDVI=computeNDVI())
    computes and returns the entire Portion of Vegetation using the mathematical expression:
    Square((“NDVI” - NDVImin) / (NDV1max - NDVImin)).

    parameters
    -------------
    NDVI: ndarray
        computed NDVI (default = computeTNDVI())

    returns
    -------------
    PV(): ndarray
        Portion of Vegetation: np.square( ( NDVI - NDVImin ) / ( NDVImax - NDVImin ) )
    """

    NDVImin=np.nanmin(NDVI) 
    NDVImax=np.nanmax(NDVI)

    return np.square( ( NDVI - NDVImin ) / ( NDVImax - NDVImin ) )



def computeEmissivity(PV=computePV(), Ev=0.986, Es=0.973, C=0.0001):
    r"""
    computeEmissivity(PV=computePV(), Ev=0.986, Es=0.973, C=0.0001) computes and returns
    the spectral Emissivity of the portion of Vegetation(PV) with the mathematical expression:
    E = EvPVRv + Es(1-PV)Rs + C, where:
    PV: is Portion of Vegetation, 
    Ev: and Es are the vegetation and soil emissivities with constant emperical values of 0.986 and 0.973
    respectively,
    Rv =(0.92762 + 0.07033PV), Rs=(0.99782 + 0.05362PV), 
    C: represents the surface roughness (topography factor) which is equal to '0' for smooth and homogeneous surface.
    For illustrative purposes, we assume C = 0.0001 for all areas.
    
    parameters
    -------------
    PV: ndarray
        computed PV (default = computeTPV())
    Ev: float
        Vegetation emissivity (default = 0.986)
    Es: float
        Soil emissivity (default = 0.973)
    C:  float
        topography factor(default = 0.0001)

    returns
    -------------
    E: ndarray
       emissivity: ( Ev * PV * Rv ) + ( Es * ( 1 - PV ) * Rs ) + C
    """

    Rv =(0.92762 + (0.07033*PV))
    Rs=(0.99782 + (0.05362*PV))
    
    #return 0.004 * PV + 0.986
    return ( Ev * PV * Rv ) + ( Es * ( 1 - PV ) * Rs ) + C



def compute_p(h=6.626e-34, c=2.998e8, a=1.38e-23):
    r"""
    compute_p(h=6.626e-34, c=2.998e8, a=1.38e-23) computes and returns the
    value of p  with the following mathematical expression p = h*c/a, where: 
    h: is the Planks constant(6.626e-34Js), 
    c: is the speed of light (2.998e8m/s),
    a: is the Boltzmann constant(1.38e-23J/K)

    parameters
    -------------
    h: float
        Planc's constant (default = 6.626e-34)
    c: float
         Speed of light (default = 2.99e8)
    a: float
        Boltzmann's constant (default = 1.38e-23)

    returns
    -------------
    p: float
       p: h * c / a
    """

    return h * c / a



def computeLST(E=computeEmissivity(), BT=computeBT(), p=compute_p(), kn=10.895e-6):
    r"""
    computeLST(E=computeEmissivity(), BT=computeBT(), p=compute_p(), kn=10.895e-6)
    computes and returns the Land Surface Temperature using the following
    mathematical expression: LST = BT/(1 + (kn * BT/p) * ln(E))-273.15
    where: BT is Brightness Temperature and E is Emissivity
    Kn is emitted radiancewavelength(center wavelength of band 10) = 10.895um = 10.895e-6m.

    parameters
    -------------
    E:  ndarray
        Emissivity (default = computeEmissivity())
    BT: ndarray
         Brightness Temperature (default = computeBT())
    p:  float
        p (default = compute_p())
    Kn: float
        Central Wavelength of TIRS band (default = 10.895e-6)

    returns
    -------------
    LST: ndarray
        LST(Land Surface Temperature): BT / ( 1 + ( ( kn * BT / p ) * np.log(E) ) )
    """
    return BT / ( 1 + ( ( kn * BT / p ) * np.log(E) ) )


def feature_min_max_compute(*features):
    r"""
    feature_min_max_compute(*features) computes the min and max values of a feature

    parameters
    -------------
    features: ndarray
        features

    returns
    -------------
    feature_min: float
        minmum value in each feature
    feature_max: float
        maximum value in each feature
    """

    return tuple((np.nanmin(feature), np.nanmax(feature)) for feature in features)



def stackFeatures(NDVI, PV, LST, transpose=True):
    r"""
    stackFeatures(NDVI, PV, LST, transpose=True) takes all three computed features as arguments and
    returns a dictionary with the following keys: ('NDVIminmax', 'PVminmax', LSTminmax, stacked_data).
    the shape of the stacked data is (W, H, C) if 'transpose == True' and (C, W, H) if transpose !=True
    where W is 'matrix width', H is 'matrix height' and C is 'features/channel'

    parameters
    -------------
    NDVI: ndarray
        computed Normalized Differential Vegetative Index (default = computeNDVI())
    PV: ndarray
        computed Portion of Vegetation (default = computePV())
    LST: ndarray
        computed Land Surface Temperature(default = computeLST())
    transpose: boolean
        transpose the array by (1, 2, 0) from (C, W, H) to (W, H, C) (default = True

    returns
    -------------
    LST: ndarray
        LST(Land Surface Temperature): BT / ( 1 + ( ( kn * BT / p ) * np.log(E) ) )
    """

    (NDVImin, NDVImax), (PVmin, PVmax), (LSTmin, LSTmax) = feature_min_max_compute(NDVI, PV, LST)
    feature_data = {
                    "NDVIminmax":(NDVImin, NDVImax),
                    "PVminmax":(PVmin, PVmax),
                    "LSTminmax":(LSTmin, LSTmax)
                }
    if transpose == True:
            feature_data["stacked_data"] = np.transpose(np.stack((NDVI, PV, LST)), (1, 2, 0))
    else:
        feature_data["stacked_data"] = np.stack((NDVI, PV, LST))

    return feature_data



def saveModel(model, dir="model", path="Kmeans_model.sav"):
    r"""
    save_model(model, dir, path) serializes the trained model as a pickle file
    for inference

    parameters
    -------------
    model: model
        the machine learning model to be saved
    dir: string
        the directory of the model path
    path: string
        the path to save the model
    
    returns
    -------------
    None
    """
    file_path = os.path.join(dir, path)
    pickle.dump(model, open(file_path, "wb"))



def saveModelOutput(feature_data, path, labels=None, dir="output", data_cols=['NDVI', 'PV', 'LST'], label_col="Cluster_label"):
    r"""
    saveModelOutput(feature_data, label, path, dir, n_features, data_cols, label_col)
    saves any output of the model in a pandas dataframe and writes it into a csv file
    on the local drive.

    parameters
    -------------
    feature_data: ndarray
        feature to be saved
    labels:  ndarray
        cluster labels of the cluster algorithm
    path: string
        name of file path
    dir: string
        directory of path (default = "output")
    data_cols: list
        names of columns of the dataframe (default = ['NDVI', 'PV', 'LST'])
    label_col:  string
        name of cluster label column (default = "Cluster_label")
    
    returns
    -------------
    None
    """

    file_path = os.path.join(dir, path)
    df = pd.DataFrame(data=feature_data, columns=data_cols)
    if isinstance(labels, (list, np.ndarray)):
        df[label_col] = labels
    df.to_csv(file_path)



def train_process(feature_data, init_centroids=None, feature_per_sample = n_features, n_clusters=60,
     fill_values=None, remove_nan=True):
    r"""
    train_process(feature_data, init_centroids=None, feature_per_sample = 3, n_clusters=4, fill_values=0.0,
    remove_nan=True).
    The Machine Learning model is based off of the K-Means Cluster algorithm which uses the
    unsupervised Learning approach of machine learning.
    This approach was implemented due to the reason that most data found about factors that affect
    occurance and spread of wildfire had no labels and so hence this approach deals with such isssue 
    by forming 'K' number of clusters (given 'K' number of centroids) from the unlabeled features.
    These cluster will futhermore be studied so as to best stipulate satisfactory condition on why a
    given set of features fell into a particular cluster and accertain a conditions for the occurance 
    and spread of wildfire and give each cluster label a value.

    parameters
    -------------
    feature_data: ndarray
        feature data used to train the cluster algorithm
    init_centroid: ndarray
        the initial centroids (default = None)
        if init_centroid is 'None' the default centroid init algorithm of the cluster will be used. 
    feature_per_sample: int
        the number of features of the training data (default = 3)
    n_clusters: int
        number of clusters for the cluster algorithm(default = 4)
    fill_values: None
        values to use inplace of nan values (default = None).
        Set the 'fill values' only when 'remove_nan' is 'False'
    remove_nan: boolean
        removes rows with nan values (default = True)
        set the 'fill_values' when this is set to 'False'

    returns
    -------------
    centroids: ndarray
        cluster_labels
    """

    feature_data = feature_data["stacked_data"].reshape(-1, feature_per_sample)
    feature_data = np.array(feature_data, dtype=np.float)
    if remove_nan == True:
        #remove nan values
        feature_data = feature_data[~np.isnan(feature_data).any(axis=1)]
    else:
        feature_data = np.nan_to_num(feature_data, nan=fill_values)

    print("training start..........")
    start_time = time.time()
    if init_centroids == None:
        clusterAlgorithm = MiniBatchKMeans(n_clusters=n_clusters)
        clusterAlgorithm.fit(feature_data)
    else:
        clusterAlgorithm = MiniBatchKMeans(n_clusters=n_clusters, init=init_centroids)
        clusterAlgorithm.fit(feature_data)    
    end_time = time.time()
    print("training ended.")
    print(f"total_training_time: {(end_time - start_time)/60}mins.")

    #save model
    saveModel(clusterAlgorithm)
    cluster_labels = clusterAlgorithm.labels_
    centroids = clusterAlgorithm.cluster_centers_

    #save features and labels in csv file
    saveModelOutput(feature_data[:50000], labels=cluster_labels[:50000], path="features.csv")
    #save centroids in csv file
    saveModelOutput(centroids, path="centroids.csv")

    return cluster_labels, centroids

feature_data = stackFeatures(computeNDVI(), computePV(), computeLST())
train_process(feature_data)