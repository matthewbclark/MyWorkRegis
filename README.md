## Using QGIS and Support Vector Machines to Differentiate Species of Meadow Jumping Mice (*Zapus hudsonius*) and Western Jumping Mice (*Zapus princeps*)
**Matthew Clark**

**Introduction**

Colorado is home to two different jumping mouse species, the meadow jumping mouse (*Zapus hudsonius*) and the western jumping mouse (*Zapus princeps*). Many of the subspecies of meadow jumping mice including the Preble's meadow jumping mouse (*Zapus hudsonius preblei*) are listed as endangered (US Fish and Wildlife Service, 2021). Conservation plans for construction along the Front Range frequently have to include plans to avoid or minimize negative effects on the jumping mice. In contrast, the western jumping mouse is thriving. This project will look into whether habitat differences between the two species can be classified by a machine learning algorithm. This will take the form of a support vector machine learner. The goal of the project is to specifically see if the habitat characteristics of meadow jumping mouse habitat can be detected by the SVM. For this reason, the sightings of meadow  jumping mice and western jumping mice will be compared to see if the model can tell the differences between them. The US Fish and Wildlife Service recommends that habitat for the Preble's meadow jumping mouse be within 110 meters of a water body, be it a stream, river, pond or lake (Trainor et. al. 2012). However, Trainor et. al. (2012) have found that some mice can be found as far as 340 meters from a body of water as the mice also frequent the grasslands in the vacinity of rivers. These parameters would be important in this project. 

**Data Sources**

Data was acquired from BISON (Biodiversity Information Serving Our Nation) in the form of point shapefiles containing data on the geographic location of the specimens, the institutions who collected the data, taxonomic information and the date in which it was collected (BISON, n.d.). All data points before 1990 were excluded as the areas where this data was found may have been developed by the present day. In addition the data was clipped to just include data points from Colorado. Institutions who contributed to the data include the Denver Museum of Nature and Science, NatureServe Network, the Museum of Southwestern Biology, Fort Hayes Sternberg Museum of Natural History, University of Alaska Museum of the North, iNaturalist.org, Angelo State Natural History Museum, Charles R. Conner Museum and the University of Colorado Museum of Natural History.

Data for land cover was retrieved from the US Geological Survey's *2011 National Land Cover Dataset* or NLDS 2011 (United States Geological Survey, 2011). This dataset contains 20 different land cover types. This study included open water, open space, developed areas (low, medium and high), barren ground, deciduous forests, coniferous forests, mixed forests, shrubs, grasslands, pasture, agricultural areas, wooded wetlands and herbaceous emergent grasslands. Data for rivers was acquired from the USGS National Geospatial Program's map, *NHD 20200615 for Colorado State or Territory SDshapefile Model Version 2.2.1* (US Geological Survey, 2020). Elevation data was collected from a dataset created by ColoradoView/UV-B Monitoring and Research (n.d.). This dataset consisted of 28 separate raster files representing a digital elevation model (DEM) for the state of Colorado.

**QGIS**

QGIS software was used to process the data needed for the project. This is the key way in which this project was different from others in that data was not collected via an API, but rather by combining external data sources using GIS software. QGIS is an open source software in comparison to ArcGIS which is very expensive. QGIS allowed the data to be processed remotely without an expensive subscription to ArcGIS. At the beginning of the project, a proof-of-concept model was created to see if it was indeed possible to create the required data on QGIS. 

All data needed to be converted to the North America Lambert Conformal Conic projection to ensure that the data was able to line up and overlap properly. A dataset of Colorado Counties was used as a mask to clip the 2011 National Land Cover Database (NLDS 2011) to just Colorado. In the case of the proof-of-concept model, this was just to Douglas County. The river data from the USGS was clipped to just include Colorado counties, as it previously included the entire watersheds in the region around Colorado. This data was composed not just of major rivers, but also the flowlines in each direction (North, South, East, Northeast, Southeast, West, Northwest and Southwest). In order to calculate the distance from streams required this data to be merged. This data was then transformed into a raster using the *rasterize* function. The finest resolution possible was a 10-square meter resolution. Distances from rivers were then calculated using 10-square meter increments using the *proximity* function.

The BISON data included sightings of both the western and meadow jumping mice, and these shapefiles needed to be converted to the Lambert Conformal Conic coordinate system and then combined using the *merge* function. Because the datasets all contained the same columns, this was actually very easy. Buffers were calculated at a distance of  340-meters as per the observations of Trainor et. al. 2012. The *zonal histogram* function was then used to perform a count of the number of pixels in each land cover type that overlapped with each buffer polygon. These values came from approximately 28 square-meter cells that would later be multiplied by 28 with Pandas to produce the area in square-meters of each habitat type within each buffer. The *zonal statistics* function was used to find the average distance value in 10x10 square meter pixels from a river within each buffer. This value would later be multiplied by 10 to estimate the average distance in square meters from rivers within the buffer. The zonal statistics function would also perform a similar operation to determine the average elevation in meters within each buffer.



### Modules used in this project

**Pandas**

**numpy**

**seaborn**

**numpy**

**IPython**

**Matplotlib**

**sklearn**
sklearn functions include train_test_split, accuracy_score, svm, metrics, cross_val_score, DecisionTreeClassifier, FactorAnalysis

**Preparing the Data**

Data preparation involved exporting the data from the final data's attribute table as a .csv file, and then using Pandas to edit the data. This involved creating a list of new names for the subsequent columns in the dataset and applying them with the .columns function. As the data was prepared in QGIS, it was not necessary to fill in columns with missing data. Lambda functions were then used to further edit the data. Some points fell outside of the scope of the NLCD and had a value of NoData in some or all of their buffers. To fix this problem, these cells were multiplied by zero in the lambda functions so that this column would not interfere with data analysis. Each column for land cover in the NLCD had counts of approximately 28 square meter cells, so each column needed to be multiplied by 28. The finest resolution possible for the distance from rivers were 10 square meter cells, so the values in this column needed to be multiplied by 10. 




<div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fid</th>
      <th>bisonID</th>
      <th>ITISsciNme</th>
      <th>xcoord</th>
      <th>ycoord</th>
      <th>nlcdb2011_0</th>
      <th>nlcdb2011_11</th>
      <th>nlcdb2011_21</th>
      <th>nlcdb2011_22</th>
      <th>nlcdb2011_23</th>
      <th>...</th>
      <th>nlcdb2011_42</th>
      <th>nlcdb2011_43</th>
      <th>nlcdb2011_52</th>
      <th>nlcdb2011_71</th>
      <th>nlcdb2011_81</th>
      <th>nlcdb2011_82</th>
      <th>nlcdb2011_90</th>
      <th>nlcdb2011_95</th>
      <th>riverdist_decameters_mean</th>
      <th>Ele_meters_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1061284029</td>
      <td>Zapus princeps</td>
      <td>-1.048927e+06</td>
      <td>34400.810906</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>263</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>18.462810</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>897076990</td>
      <td>Zapus princeps</td>
      <td>-8.852860e+05</td>
      <td>161407.789815</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>0</td>
      <td>116</td>
      <td>0</td>
      <td>132</td>
      <td>0</td>
      <td>87</td>
      <td>0</td>
      <td>13.076882</td>
      <td>1362.505689</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>897077009</td>
      <td>Zapus princeps</td>
      <td>-8.852860e+05</td>
      <td>161407.789815</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>0</td>
      <td>116</td>
      <td>0</td>
      <td>132</td>
      <td>0</td>
      <td>87</td>
      <td>0</td>
      <td>13.076882</td>
      <td>1362.505689</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>897077028</td>
      <td>Zapus princeps</td>
      <td>-8.852860e+05</td>
      <td>161407.789815</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>0</td>
      <td>116</td>
      <td>0</td>
      <td>132</td>
      <td>0</td>
      <td>87</td>
      <td>0</td>
      <td>13.076882</td>
      <td>1362.505689</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1837315390</td>
      <td>Zapus princeps</td>
      <td>-8.850253e+05</td>
      <td>161374.025382</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>54</td>
      <td>0</td>
      <td>83</td>
      <td>0</td>
      <td>20.020843</td>
      <td>1372.888781</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>397</th>
      <td>398</td>
      <td>1145109121</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>247</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>0</td>
      <td>13.668359</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>398</th>
      <td>399</td>
      <td>1145109130</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>247</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>0</td>
      <td>13.668359</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>399</th>
      <td>400</td>
      <td>1145109134</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>247</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>0</td>
      <td>13.668359</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>400</th>
      <td>401</td>
      <td>1145109154</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>247</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>0</td>
      <td>13.668359</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>401</th>
      <td>402</td>
      <td>1145109161</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>247</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>0</td>
      <td>13.668359</td>
      <td>1761.247582</td>
    </tr>
  </tbody>
</table>
<p>402 rows × 23 columns</p>
</div>

<div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fid</th>
      <th>bisonID</th>
      <th>Species</th>
      <th>xcoord</th>
      <th>ycoord</th>
      <th>NoData</th>
      <th>Open_water</th>
      <th>Dev_open_space</th>
      <th>Dev_low</th>
      <th>Dev_medium</th>
      <th>...</th>
      <th>Conifer_forest</th>
      <th>Mixed_forest</th>
      <th>Shrubland</th>
      <th>Grassland</th>
      <th>Pasture</th>
      <th>Agriculture</th>
      <th>Wetlands_woody</th>
      <th>Wetlands_herb</th>
      <th>River_Distance</th>
      <th>Elevation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1061284029</td>
      <td>Zapus princeps</td>
      <td>-1.048927e+06</td>
      <td>34400.810906</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>263</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>18.462810</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>897076990</td>
      <td>Zapus princeps</td>
      <td>-8.852860e+05</td>
      <td>161407.789815</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>0</td>
      <td>116</td>
      <td>0</td>
      <td>132</td>
      <td>0</td>
      <td>87</td>
      <td>0</td>
      <td>13.076882</td>
      <td>1362.505689</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>897077009</td>
      <td>Zapus princeps</td>
      <td>-8.852860e+05</td>
      <td>161407.789815</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>0</td>
      <td>116</td>
      <td>0</td>
      <td>132</td>
      <td>0</td>
      <td>87</td>
      <td>0</td>
      <td>13.076882</td>
      <td>1362.505689</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>897077028</td>
      <td>Zapus princeps</td>
      <td>-8.852860e+05</td>
      <td>161407.789815</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>0</td>
      <td>116</td>
      <td>0</td>
      <td>132</td>
      <td>0</td>
      <td>87</td>
      <td>0</td>
      <td>13.076882</td>
      <td>1362.505689</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1837315390</td>
      <td>Zapus princeps</td>
      <td>-8.850253e+05</td>
      <td>161374.025382</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>54</td>
      <td>0</td>
      <td>83</td>
      <td>0</td>
      <td>20.020843</td>
      <td>1372.888781</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>397</th>
      <td>398</td>
      <td>1145109121</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>247</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>0</td>
      <td>13.668359</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>398</th>
      <td>399</td>
      <td>1145109130</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>247</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>0</td>
      <td>13.668359</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>399</th>
      <td>400</td>
      <td>1145109134</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>247</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>0</td>
      <td>13.668359</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>400</th>
      <td>401</td>
      <td>1145109154</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>247</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>0</td>
      <td>13.668359</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>401</th>
      <td>402</td>
      <td>1145109161</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>247</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>0</td>
      <td>13.668359</td>
      <td>1761.247582</td>
    </tr>
  </tbody>
</table>
<p>402 rows × 23 columns</p>
</div>


<div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fid</th>
      <th>bisonID</th>
      <th>Species</th>
      <th>xcoord</th>
      <th>ycoord</th>
      <th>NoData</th>
      <th>Open_water</th>
      <th>Dev_open_space</th>
      <th>Dev_low</th>
      <th>Dev_medium</th>
      <th>...</th>
      <th>Conifer_forest</th>
      <th>Mixed_forest</th>
      <th>Shrubland</th>
      <th>Grassland</th>
      <th>Pasture</th>
      <th>Agriculture</th>
      <th>Wetlands_woody</th>
      <th>Wetlands_herb</th>
      <th>River_Distance</th>
      <th>Elevation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1061284029</td>
      <td>Zapus princeps</td>
      <td>-1.048927e+06</td>
      <td>34400.810906</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7364</td>
      <td>0</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>184.628104</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>897076990</td>
      <td>Zapus princeps</td>
      <td>-8.852860e+05</td>
      <td>161407.789815</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>336</td>
      <td>0</td>
      <td>3248</td>
      <td>0</td>
      <td>3696</td>
      <td>0</td>
      <td>2436</td>
      <td>0</td>
      <td>130.768821</td>
      <td>1362.505689</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>897077009</td>
      <td>Zapus princeps</td>
      <td>-8.852860e+05</td>
      <td>161407.789815</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>336</td>
      <td>0</td>
      <td>3248</td>
      <td>0</td>
      <td>3696</td>
      <td>0</td>
      <td>2436</td>
      <td>0</td>
      <td>130.768821</td>
      <td>1362.505689</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>897077028</td>
      <td>Zapus princeps</td>
      <td>-8.852860e+05</td>
      <td>161407.789815</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>336</td>
      <td>0</td>
      <td>3248</td>
      <td>0</td>
      <td>3696</td>
      <td>0</td>
      <td>2436</td>
      <td>0</td>
      <td>130.768821</td>
      <td>1362.505689</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1837315390</td>
      <td>Zapus princeps</td>
      <td>-8.850253e+05</td>
      <td>161374.025382</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>616</td>
      <td>0</td>
      <td>5236</td>
      <td>0</td>
      <td>1512</td>
      <td>0</td>
      <td>2324</td>
      <td>0</td>
      <td>200.208428</td>
      <td>1372.888781</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>397</th>
      <td>398</td>
      <td>1145109121</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
      <td>136.683588</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>398</th>
      <td>399</td>
      <td>1145109130</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
      <td>136.683588</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>399</th>
      <td>400</td>
      <td>1145109134</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
      <td>136.683588</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>400</th>
      <td>401</td>
      <td>1145109154</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
      <td>136.683588</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>401</th>
      <td>402</td>
      <td>1145109161</td>
      <td>Zapus hudsonius</td>
      <td>-7.269401e+05</td>
      <td>-19090.725997</td>
      <td>0</td>
      <td>0</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
      <td>136.683588</td>
      <td>1761.247582</td>
    </tr>
  </tbody>
</table>
<p>402 rows × 23 columns</p>
</div>



**Coding for Data Exploration**

Data exploration was done with the Seaborn, Matplotlib and IPython modules. Creating these plots relied on seperating numerical data for land cover types, elevations and river distances using the .iloc[] function from Pandas. The pair plot was created Seaborn's pairplot() function. In contrast, the heat map involved converting the dataframe into an array, and then using Seaborn's heatmap function combined with a corr() function using techniques from Anita (2019). 

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Species</th>
      <th>Dev_open_space</th>
      <th>Dev_low</th>
      <th>Dev_medium</th>
      <th>Dev_high</th>
      <th>Barren_land</th>
      <th>Deci_forest</th>
      <th>Conifer_forest</th>
      <th>Mixed_forest</th>
      <th>Shrubland</th>
      <th>Grassland</th>
      <th>Pasture</th>
      <th>Agriculture</th>
      <th>Wetlands_woody</th>
      <th>Wetlands_herb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Zapus princeps</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3948</td>
      <td>7364</td>
      <td>0</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Zapus princeps</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2156</td>
      <td>336</td>
      <td>0</td>
      <td>3248</td>
      <td>0</td>
      <td>3696</td>
      <td>0</td>
      <td>2436</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Zapus princeps</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2156</td>
      <td>336</td>
      <td>0</td>
      <td>3248</td>
      <td>0</td>
      <td>3696</td>
      <td>0</td>
      <td>2436</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Zapus princeps</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2156</td>
      <td>336</td>
      <td>0</td>
      <td>3248</td>
      <td>0</td>
      <td>3696</td>
      <td>0</td>
      <td>2436</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Zapus princeps</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2324</td>
      <td>616</td>
      <td>0</td>
      <td>5236</td>
      <td>0</td>
      <td>1512</td>
      <td>0</td>
      <td>2324</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>397</th>
      <td>Zapus hudsonius</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
    </tr>
    <tr>
      <th>398</th>
      <td>Zapus hudsonius</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
    </tr>
    <tr>
      <th>399</th>
      <td>Zapus hudsonius</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
    </tr>
    <tr>
      <th>400</th>
      <td>Zapus hudsonius</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
    </tr>
    <tr>
      <th>401</th>
      <td>Zapus hudsonius</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>402 rows × 15 columns</p>
</div>


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open_water</th>
      <th>Dev_open_space</th>
      <th>Dev_low</th>
      <th>Dev_medium</th>
      <th>Dev_high</th>
      <th>Barren_land</th>
      <th>Deci_forest</th>
      <th>Conifer_forest</th>
      <th>Mixed_forest</th>
      <th>Shrubland</th>
      <th>Grassland</th>
      <th>Pasture</th>
      <th>Agriculture</th>
      <th>Wetlands_woody</th>
      <th>Wetlands_herb</th>
      <th>River_Distance</th>
      <th>Elevation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3948</td>
      <td>7364</td>
      <td>0</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>184.628104</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2156</td>
      <td>336</td>
      <td>0</td>
      <td>3248</td>
      <td>0</td>
      <td>3696</td>
      <td>0</td>
      <td>2436</td>
      <td>0</td>
      <td>130.768821</td>
      <td>1362.505689</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2156</td>
      <td>336</td>
      <td>0</td>
      <td>3248</td>
      <td>0</td>
      <td>3696</td>
      <td>0</td>
      <td>2436</td>
      <td>0</td>
      <td>130.768821</td>
      <td>1362.505689</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2156</td>
      <td>336</td>
      <td>0</td>
      <td>3248</td>
      <td>0</td>
      <td>3696</td>
      <td>0</td>
      <td>2436</td>
      <td>0</td>
      <td>130.768821</td>
      <td>1362.505689</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2324</td>
      <td>616</td>
      <td>0</td>
      <td>5236</td>
      <td>0</td>
      <td>1512</td>
      <td>0</td>
      <td>2324</td>
      <td>0</td>
      <td>200.208428</td>
      <td>1372.888781</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>397</th>
      <td>0</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
      <td>136.683588</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>398</th>
      <td>0</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
      <td>136.683588</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>399</th>
      <td>0</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
      <td>136.683588</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>400</th>
      <td>0</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
      <td>136.683588</td>
      <td>1761.247582</td>
    </tr>
    <tr>
      <th>401</th>
      <td>0</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
      <td>136.683588</td>
      <td>1761.247582</td>
    </tr>
  </tbody>
</table>
<p>402 rows × 17 columns</p>
</div>



**Results for Data Explortation**

The shear number of variables made interpreting the pairplot difficult. Many variables had skewed distributions due to the high number of zero values. It is hoped that the large sample sizes will be sufficient to balance for this effect. Strong relationships between the different development types were detected in both the correlation matrix with a heatmap and with the pairplot. The heat map in particular showed a strong relationship between open space development and low developed areas, medium development and high development areas and between river distance and grasslands. The final correlation is likely due to the fact that many Preble's jumping mouse sightings were found in grasslands within 320 meters from water, as observed in Trainor et. al. (2012). A correlation between mixed forest and decidous forest was also observed. Coniferous forest correlated heavily in a positive way to elevation and strongly in a negative way to grasslands. This is not surprising as the two habitat types are found at different elevations. 
    
![png](output_21_0.png)
    


    <AxesSubplot:>




    
![png](output_23_1.png)
    


<div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open_water</th>
      <th>Dev_open_space</th>
      <th>Dev_low</th>
      <th>Dev_medium</th>
      <th>Dev_high</th>
      <th>Barren_land</th>
      <th>Deci_forest</th>
      <th>Conifer_forest</th>
      <th>Mixed_forest</th>
      <th>Shrubland</th>
      <th>Grassland</th>
      <th>Pasture</th>
      <th>Agriculture</th>
      <th>Wetlands_woody</th>
      <th>Wetlands_herb</th>
      <th>River_Distance</th>
      <th>Elevation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Open_water</th>
      <td>1.000000</td>
      <td>-0.036538</td>
      <td>0.021182</td>
      <td>0.004826</td>
      <td>-0.006283</td>
      <td>0.144318</td>
      <td>-0.018428</td>
      <td>-0.020311</td>
      <td>-0.011812</td>
      <td>-0.080906</td>
      <td>-0.045913</td>
      <td>0.028194</td>
      <td>0.095883</td>
      <td>0.135825</td>
      <td>-0.016493</td>
      <td>-0.056878</td>
      <td>0.042670</td>
    </tr>
    <tr>
      <th>Dev_open_space</th>
      <td>-0.036538</td>
      <td>1.000000</td>
      <td>0.569501</td>
      <td>0.375584</td>
      <td>0.005372</td>
      <td>0.092142</td>
      <td>-0.240566</td>
      <td>-0.154506</td>
      <td>-0.166893</td>
      <td>-0.188156</td>
      <td>-0.055796</td>
      <td>0.069983</td>
      <td>0.062039</td>
      <td>0.216518</td>
      <td>0.440834</td>
      <td>-0.166596</td>
      <td>-0.155284</td>
    </tr>
    <tr>
      <th>Dev_low</th>
      <td>0.021182</td>
      <td>0.569501</td>
      <td>1.000000</td>
      <td>0.679961</td>
      <td>0.145106</td>
      <td>-0.019672</td>
      <td>-0.177684</td>
      <td>-0.250567</td>
      <td>-0.100186</td>
      <td>-0.110592</td>
      <td>-0.061569</td>
      <td>0.009865</td>
      <td>0.235022</td>
      <td>0.081827</td>
      <td>0.390990</td>
      <td>-0.125091</td>
      <td>-0.210519</td>
    </tr>
    <tr>
      <th>Dev_medium</th>
      <td>0.004826</td>
      <td>0.375584</td>
      <td>0.679961</td>
      <td>1.000000</td>
      <td>0.644879</td>
      <td>-0.005309</td>
      <td>-0.135071</td>
      <td>-0.198269</td>
      <td>-0.075286</td>
      <td>-0.132820</td>
      <td>-0.105259</td>
      <td>0.006409</td>
      <td>0.055239</td>
      <td>0.044841</td>
      <td>0.170482</td>
      <td>-0.044876</td>
      <td>-0.150872</td>
    </tr>
    <tr>
      <th>Dev_high</th>
      <td>-0.006283</td>
      <td>0.005372</td>
      <td>0.145106</td>
      <td>0.644879</td>
      <td>1.000000</td>
      <td>-0.010526</td>
      <td>-0.058957</td>
      <td>-0.086835</td>
      <td>-0.029668</td>
      <td>-0.089826</td>
      <td>-0.079595</td>
      <td>-0.025843</td>
      <td>-0.000749</td>
      <td>-0.041685</td>
      <td>-0.030765</td>
      <td>0.023966</td>
      <td>-0.072088</td>
    </tr>
    <tr>
      <th>Barren_land</th>
      <td>0.144318</td>
      <td>0.092142</td>
      <td>-0.019672</td>
      <td>-0.005309</td>
      <td>-0.010526</td>
      <td>1.000000</td>
      <td>-0.035945</td>
      <td>0.050467</td>
      <td>-0.029884</td>
      <td>-0.096226</td>
      <td>-0.048178</td>
      <td>-0.027834</td>
      <td>-0.007122</td>
      <td>-0.009112</td>
      <td>-0.034465</td>
      <td>-0.050859</td>
      <td>0.204099</td>
    </tr>
    <tr>
      <th>Deci_forest</th>
      <td>-0.018428</td>
      <td>-0.240566</td>
      <td>-0.177684</td>
      <td>-0.135071</td>
      <td>-0.058957</td>
      <td>-0.035945</td>
      <td>1.000000</td>
      <td>-0.079970</td>
      <td>0.506314</td>
      <td>-0.080500</td>
      <td>-0.344262</td>
      <td>-0.015278</td>
      <td>-0.036849</td>
      <td>-0.101884</td>
      <td>-0.192160</td>
      <td>-0.211640</td>
      <td>0.290375</td>
    </tr>
    <tr>
      <th>Conifer_forest</th>
      <td>-0.020311</td>
      <td>-0.154506</td>
      <td>-0.250567</td>
      <td>-0.198269</td>
      <td>-0.086835</td>
      <td>0.050467</td>
      <td>-0.079970</td>
      <td>1.000000</td>
      <td>0.046850</td>
      <td>-0.105764</td>
      <td>-0.546043</td>
      <td>-0.135393</td>
      <td>-0.056068</td>
      <td>-0.067948</td>
      <td>-0.283852</td>
      <td>-0.167159</td>
      <td>0.515689</td>
    </tr>
    <tr>
      <th>Mixed_forest</th>
      <td>-0.011812</td>
      <td>-0.166893</td>
      <td>-0.100186</td>
      <td>-0.075286</td>
      <td>-0.029668</td>
      <td>-0.029884</td>
      <td>0.506314</td>
      <td>0.046850</td>
      <td>1.000000</td>
      <td>-0.188500</td>
      <td>-0.229213</td>
      <td>0.010897</td>
      <td>-0.018883</td>
      <td>-0.091582</td>
      <td>-0.096215</td>
      <td>-0.123846</td>
      <td>0.236488</td>
    </tr>
    <tr>
      <th>Shrubland</th>
      <td>-0.080906</td>
      <td>-0.188156</td>
      <td>-0.110592</td>
      <td>-0.132820</td>
      <td>-0.089826</td>
      <td>-0.096226</td>
      <td>-0.080500</td>
      <td>-0.105764</td>
      <td>-0.188500</td>
      <td>1.000000</td>
      <td>-0.289694</td>
      <td>-0.073769</td>
      <td>-0.058049</td>
      <td>-0.199858</td>
      <td>-0.132898</td>
      <td>-0.380409</td>
      <td>-0.333746</td>
    </tr>
    <tr>
      <th>Grassland</th>
      <td>-0.045913</td>
      <td>-0.055796</td>
      <td>-0.061569</td>
      <td>-0.105259</td>
      <td>-0.079595</td>
      <td>-0.048178</td>
      <td>-0.344262</td>
      <td>-0.546043</td>
      <td>-0.229213</td>
      <td>-0.289694</td>
      <td>1.000000</td>
      <td>-0.182204</td>
      <td>-0.030132</td>
      <td>-0.168646</td>
      <td>0.119018</td>
      <td>0.680129</td>
      <td>-0.261759</td>
    </tr>
    <tr>
      <th>Pasture</th>
      <td>0.028194</td>
      <td>0.069983</td>
      <td>0.009865</td>
      <td>0.006409</td>
      <td>-0.025843</td>
      <td>-0.027834</td>
      <td>-0.015278</td>
      <td>-0.135393</td>
      <td>0.010897</td>
      <td>-0.073769</td>
      <td>-0.182204</td>
      <td>1.000000</td>
      <td>-0.008852</td>
      <td>0.304479</td>
      <td>-0.016719</td>
      <td>-0.094673</td>
      <td>-0.129430</td>
    </tr>
    <tr>
      <th>Agriculture</th>
      <td>0.095883</td>
      <td>0.062039</td>
      <td>0.235022</td>
      <td>0.055239</td>
      <td>-0.000749</td>
      <td>-0.007122</td>
      <td>-0.036849</td>
      <td>-0.056068</td>
      <td>-0.018883</td>
      <td>-0.058049</td>
      <td>-0.030132</td>
      <td>-0.008852</td>
      <td>1.000000</td>
      <td>0.172482</td>
      <td>0.010176</td>
      <td>-0.041043</td>
      <td>-0.076413</td>
    </tr>
    <tr>
      <th>Wetlands_woody</th>
      <td>0.135825</td>
      <td>0.216518</td>
      <td>0.081827</td>
      <td>0.044841</td>
      <td>-0.041685</td>
      <td>-0.009112</td>
      <td>-0.101884</td>
      <td>-0.067948</td>
      <td>-0.091582</td>
      <td>-0.199858</td>
      <td>-0.168646</td>
      <td>0.304479</td>
      <td>0.172482</td>
      <td>1.000000</td>
      <td>0.161338</td>
      <td>-0.203844</td>
      <td>-0.079098</td>
    </tr>
    <tr>
      <th>Wetlands_herb</th>
      <td>-0.016493</td>
      <td>0.440834</td>
      <td>0.390990</td>
      <td>0.170482</td>
      <td>-0.030765</td>
      <td>-0.034465</td>
      <td>-0.192160</td>
      <td>-0.283852</td>
      <td>-0.096215</td>
      <td>-0.132898</td>
      <td>0.119018</td>
      <td>-0.016719</td>
      <td>0.010176</td>
      <td>0.161338</td>
      <td>1.000000</td>
      <td>-0.185898</td>
      <td>-0.181274</td>
    </tr>
    <tr>
      <th>River_Distance</th>
      <td>-0.056878</td>
      <td>-0.166596</td>
      <td>-0.125091</td>
      <td>-0.044876</td>
      <td>0.023966</td>
      <td>-0.050859</td>
      <td>-0.211640</td>
      <td>-0.167159</td>
      <td>-0.123846</td>
      <td>-0.380409</td>
      <td>0.680129</td>
      <td>-0.094673</td>
      <td>-0.041043</td>
      <td>-0.203844</td>
      <td>-0.185898</td>
      <td>1.000000</td>
      <td>-0.186060</td>
    </tr>
    <tr>
      <th>Elevation</th>
      <td>0.042670</td>
      <td>-0.155284</td>
      <td>-0.210519</td>
      <td>-0.150872</td>
      <td>-0.072088</td>
      <td>0.204099</td>
      <td>0.290375</td>
      <td>0.515689</td>
      <td>0.236488</td>
      <td>-0.333746</td>
      <td>-0.261759</td>
      <td>-0.129430</td>
      <td>-0.076413</td>
      <td>-0.079098</td>
      <td>-0.181274</td>
      <td>-0.186060</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Creating the SVM model and Displaying the Results**

The actual SVM model was created using code from the YouTube channel CMS WisCon (April 30, 2020). This source was chosen because this was the first time I created an SVM using Python, and I was having difficulty creating the model from a Pandas dataframe. The first step involved creating a random seed for the model using numpy's random.seed() function. Then the model was converted to an array using the .values function. Testing and training datasets were created from the original dataframe. The training data including values for the land area in square meters of different habitat types, the distance in rivers in meters and the elevation of the habitat. Latitude and longitude were left out due to the great difference in orders of magnitude of the data. The colunm for species was used for the dependent variable as it contained the species of the mouse included in the data point. The train_test_split() function from sklearn was used to split both the independent variable and the labels for the dependent variable into train and testing sets. The SVC() from sklearn was used to construct the support vector machine. Versions of the SVM were also created with linear, polynomial and radial basis function kernals. The models were fit with the .fit function, and testd using the .predict() function. The accuracy of the model was printed using the accuracy_score() function. The results of the data were displayed with a confusion matrix created from code by website Edpresso (2021) and sklearn's metrics package. These included a confusion matrix (metrics.confusion_matrix) and a classification report with precision, recall, f1-scores and support (metrics.classification_report).  Results from the support vectors machines were graphed in dataframes created using Python and R Tips (2018).

**Generic SVM**

The first model to be tested was a generic SVM model with default settings. The generic SVM model produced an accuracy value of 0.893. Overall, the model had a precision of 0.89 and a recall of 0.88. The model had a precision of 0.90 for meadow jumping mice (*Zapus hudsonius*) and 0.89 for western jumping mice (*Zapus princeps*). There were 8 western jumping mice misclassified as meadow jumping mice (false negatives) and 5 meadow jumping mice misclassified as western jumping mice (false positives). This resulted in a lower recall score for western jumping mice (0.83) in comparison to meadow jumping mice (0.93). The generic SVM model produced a Cohen's kappa score of 0.7712, indicating a good agreement between the predicted values and the true values (Lantz, 2015, pg. 323).   


<div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Z.hudsonius</th>
      <th>Z.princeps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>69</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>macro_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy</td>
      <td>0.893</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Precision</td>
      <td>0.890</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recall</td>
      <td>0.880</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f1-score</td>
      <td>0.890</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kappa</td>
      <td>0.770</td>
    </tr>
  </tbody>
</table>
</div>



**Linear Kernel SVM**

The SVM with a linear kernel produced a considerably lower accuracy with a value of 0.71. This model produced a precision value of 0.73 and a recall value. This model misclassified 28 meadow jumping mice as western jumping mice, and 7 western jumping mice as meadow jumping mice. The model had a precision value of 0.87 for meadow jumping mice and a value of 0.59 for western jumping mice. This model produced a recall value of 0.62 for meadow jumping mice and a value of 0.85 for western jumping mice. The linear SVM model produced a Cohen's kappa score of 0.4371, indicating only a moderate agreement between the predicted values and the true values (Lantz, 2015, pg. 323).  

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Z.hudsonius</th>
      <th>Z.princeps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>macro_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy</td>
      <td>0.711</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Precision</td>
      <td>0.730</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recall</td>
      <td>0.740</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f1-score</td>
      <td>0.710</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kappa</td>
      <td>0.437</td>
    </tr>
  </tbody>
</table>
</div>



**Polynomial Kernel SVM**

The SVM with a polynomial kernel produced an accuracy of 0.893, a precision value of 0.92 and a recall of 0.87. The model produced a precision of 0.86 for meadow jumping mice and a precision of 0.97 for western jumping mice. Recall values were 0.99 for meadow jumping mice and 0.74. This model misclassified 12 western jumping mice as meadow jumping mice and 1 meadow jumping mouse as a western jumping mouse. The polynomial SVM model produced a Cohen's kappa score of 0.7638, indicating a good agreement between the predicted values and the true values (Lantz, 2015, pg. 323).   
   
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Z.hudsonius</th>
      <th>Z.princeps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>73</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>macro_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy</td>
      <td>0.893</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Precision</td>
      <td>0.920</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recall</td>
      <td>0.870</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f1-score</td>
      <td>0.880</td>
    </tr>
    <tr>
      <th>4</th>
      <td>kappa</td>
      <td>0.764</td>
    </tr>
  </tbody>
</table>
</div>



**SVM with a Radial Basis Function**

This model produced an accuracy of 0.893, and an average precision of 0.89, and an average weighted precision of 0.88. Precision values for meadow jumping mice are 0.90 and western jumping mice are 0.89. Recall values were 0.93 for meadow jumping mice and 0.83 for western jumping mice. This model misclassified 8 western jumping mice as meadow jumping mice, and it misclassified 5 meadow jumping mice as western jumping mice. The radial-basis SVM model produced a Cohen's kappa score of 0.7712, indicating a good agreement between the predicted values and the true values (Lantz, 2015, pg. 323).  

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Z.hudsonius</th>
      <th>Z.princeps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>69</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>macro_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy</td>
      <td>0.893</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Precision</td>
      <td>0.890</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recall</td>
      <td>0.880</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f1-score</td>
      <td>0.890</td>
    </tr>
    <tr>
      <th>4</th>
      <td>kappa</td>
      <td>0.770</td>
    </tr>
  </tbody>
</table>
</div>



**Removing variables with high correlation from the Model**

This version of the model removed several variables with a high amount of correlation. Low development had a high degree of correlation with medium development. These variables included low development, medium development, mixed forest, elevation and river distance. It turned out that elevation was associated with different habitats, with low elevations being associated with grasslands and high elevations being associated with conifer forests. As a result, elevation was not necessary. Surprisingly grasslands were strongly correlated with river distance as many meadow jumping mouse sightings were located near rivers. 

**Generic SVM with Fewer Variables**

This model used the generic SVM, but dropped the river distance, elevation, low development, medium development and mixed forest variables in an attempt to make a less complicated model. This model ended up producing identical results to the generic model with all attributes intact. Its accuracy was 0.8903, its precision was 0.89 and its recall was 0.88.This model produced a Cohen's kappa score of 0.7712, indicating a good agreement between the predicted values and the true values (Lantz, 2015, pg. 323).  

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Species</th>
      <th>Dev_open_space</th>
      <th>Dev_high</th>
      <th>Barren_land</th>
      <th>Deci_forest</th>
      <th>Conifer_forest</th>
      <th>Shrubland</th>
      <th>Grassland</th>
      <th>Pasture</th>
      <th>Agriculture</th>
      <th>Wetlands_woody</th>
      <th>Wetlands_herb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Zapus princeps</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3948</td>
      <td>7364</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Zapus princeps</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2156</td>
      <td>336</td>
      <td>3248</td>
      <td>0</td>
      <td>3696</td>
      <td>0</td>
      <td>2436</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Zapus princeps</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2156</td>
      <td>336</td>
      <td>3248</td>
      <td>0</td>
      <td>3696</td>
      <td>0</td>
      <td>2436</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Zapus princeps</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2156</td>
      <td>336</td>
      <td>3248</td>
      <td>0</td>
      <td>3696</td>
      <td>0</td>
      <td>2436</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Zapus princeps</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2324</td>
      <td>616</td>
      <td>5236</td>
      <td>0</td>
      <td>1512</td>
      <td>0</td>
      <td>2324</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>397</th>
      <td>Zapus hudsonius</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
    </tr>
    <tr>
      <th>398</th>
      <td>Zapus hudsonius</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
    </tr>
    <tr>
      <th>399</th>
      <td>Zapus hudsonius</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
    </tr>
    <tr>
      <th>400</th>
      <td>Zapus hudsonius</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
    </tr>
    <tr>
      <th>401</th>
      <td>Zapus hudsonius</td>
      <td>1008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2688</td>
      <td>6916</td>
      <td>0</td>
      <td>0</td>
      <td>1624</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>402 rows × 12 columns</p>
</div>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Z.hudsonius</th>
      <th>Z.princeps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>69</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>macro_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy</td>
      <td>0.893</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Precision</td>
      <td>0.890</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recall</td>
      <td>0.880</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f1-score</td>
      <td>0.890</td>
    </tr>
    <tr>
      <th>4</th>
      <td>kappa</td>
      <td>0.771</td>
    </tr>
  </tbody>
</table>
</div>



**Polynomial SVM with Fewer Variables**

This model was similar to the generic SVM with fewer variables, but it used a polynomial kernel. This model misclassified 7 meadow jumping mice as western jumping mice, and it misclassified 9 western jumping mice as meadow jumping mice. The model's total accuracy was 0.868, with a precision value of 0.86 and a recall value of 0.86. The model produced a precision of 0.88 for the meadow jumping mouse and 0.84 for the western jumping mouse. It produced a recall value of 0.91 for the meadow jumping mouse and a recall of 0.81 for the western jumping mouse. The SVM model produced a Cohen's kappa score of 0.720, indicating a good agreement between the predicted values and the true values (Lantz, 2015, pg. 323).  


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Z.hudsonius</th>
      <th>Z.princeps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>macro_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy</td>
      <td>0.868</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Precision</td>
      <td>0.860</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recall</td>
      <td>0.860</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f1-score</td>
      <td>0.860</td>
    </tr>
    <tr>
      <th>4</th>
      <td>kappa</td>
      <td>0.720</td>
    </tr>
  </tbody>
</table>
</div>



**Decision Tree Models**

In addition to the SVM models, a decision tree was also created as these algorithms tend to be good at dealing with unbalanced (Boyle 2019). The actual decision tree model was created using code from Raschka and Mirjalili (2019) pg. 96. The graphic for the decision tree was created using code from Ptonski (2020). 

**Decision Tree**

The final model was a decision tree model with a Gini index and a max depth of 4. This model produced an accuracy of 0.901, with a precision of 0.93 and a recall of 0.90. For meadow jumping mice the model produced a precision of 0.89, a recall of 0.99 and misclassified 1 meadow jumping mouse as a western jumping mouse. For western jumping mice, the model produced a precision of 0.97 and a recall value of 0.81. It misclassified 9 western jumping mice as meadow jumping mice. The decision tree model produced a Cohen's kappa score of 0.820, indicating a good agreement between the predicted values and the true values (Lantz, 2015, pg. 323). Coniferous forests proved to be an important factor in classifying the mice. This makes sense as the western jumping mouse is more likely to be found in coniferous forests. Surprisingly, shrublands and barren land were also important factors in classifying the mice. 

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Z.hudsonius</th>
      <th>Z.princeps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>73</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>macro_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy</td>
      <td>0.901</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Precision</td>
      <td>0.930</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recall</td>
      <td>0.900</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f1-score</td>
      <td>0.910</td>
    </tr>
    <tr>
      <th>4</th>
      <td>kappa</td>
      <td>0.820</td>
    </tr>
  </tbody>
</table>
</div>

    
![png](output_90_0.png)
    


### Results

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>True Positives</th>
      <th>False Positives</th>
      <th>False Negatives</th>
      <th>True Negatives</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-1-Score</th>
      <th>Kappa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Generic</td>
      <td>69</td>
      <td>8</td>
      <td>5</td>
      <td>39</td>
      <td>0.893</td>
      <td>0.89</td>
      <td>0.88</td>
      <td>0.89</td>
      <td>0.772</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Linear</td>
      <td>46</td>
      <td>7</td>
      <td>28</td>
      <td>40</td>
      <td>0.710</td>
      <td>0.73</td>
      <td>0.74</td>
      <td>0.71</td>
      <td>0.437</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Polynomial</td>
      <td>73</td>
      <td>12</td>
      <td>1</td>
      <td>35</td>
      <td>0.893</td>
      <td>0.92</td>
      <td>0.87</td>
      <td>0.88</td>
      <td>0.764</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Radial-Basis Function</td>
      <td>69</td>
      <td>8</td>
      <td>5</td>
      <td>39</td>
      <td>0.893</td>
      <td>0.89</td>
      <td>0.88</td>
      <td>0.89</td>
      <td>0.771</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Reduced</td>
      <td>69</td>
      <td>8</td>
      <td>5</td>
      <td>39</td>
      <td>0.893</td>
      <td>0.89</td>
      <td>0.88</td>
      <td>0.89</td>
      <td>0.771</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Reduced Polynomial</td>
      <td>67</td>
      <td>9</td>
      <td>7</td>
      <td>38</td>
      <td>0.868</td>
      <td>0.86</td>
      <td>0.86</td>
      <td>0.86</td>
      <td>0.720</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Decision Tree</td>
      <td>73</td>
      <td>9</td>
      <td>1</td>
      <td>38</td>
      <td>0.901</td>
      <td>0.93</td>
      <td>0.90</td>
      <td>0.91</td>
      <td>0.820</td>
    </tr>
  </tbody>
</table>
</div>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Precision_hudsonius</th>
      <th>Precision_princeps</th>
      <th>Recall_hudsonius</th>
      <th>Recall_princeps</th>
      <th>F-1-Score_hudsonius</th>
      <th>F-1-Score_princeps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Generic SVM</td>
      <td>0.90</td>
      <td>0.89</td>
      <td>0.93</td>
      <td>0.83</td>
      <td>0.91</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Linear SVM</td>
      <td>0.87</td>
      <td>0.59</td>
      <td>0.62</td>
      <td>0.85</td>
      <td>0.72</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Polynomial SVM</td>
      <td>0.86</td>
      <td>0.97</td>
      <td>0.99</td>
      <td>0.74</td>
      <td>0.92</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Radial-Basis Function SVM</td>
      <td>0.90</td>
      <td>0.89</td>
      <td>0.93</td>
      <td>0.83</td>
      <td>0.91</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Reduced Generic SVM</td>
      <td>0.90</td>
      <td>0.89</td>
      <td>0.93</td>
      <td>0.83</td>
      <td>0.91</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Reduced Polynomial SVM</td>
      <td>0.88</td>
      <td>0.84</td>
      <td>0.91</td>
      <td>0.81</td>
      <td>0.89</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Decision Tree</td>
      <td>0.89</td>
      <td>0.97</td>
      <td>0.99</td>
      <td>0.81</td>
      <td>0.94</td>
      <td>0.88</td>
    </tr>
  </tbody>
</table>
</div>


**Comparing Models**

The model with th highest F1-score was the decision tree model with 0.91. The generic SVM, reduced SVM and radial-basis functions all produced values of 0.89. The polynomial model produced a value of 0.88. The model with the lowest F1-score was the linear model with 0.71. The next lowest was the polynomial SVM with a reduced number of variables. This was a value of 0.86.


**Discussion**

Based on F1 scores, the best model was the decision tree model, but generic model, the generic model with reduced variables and the radial-basis function were not far behind with 0.89. Overall there wasn't a large difference between models except for the linear model. The linear model performed the worst with 0.71, which is not surprising considering the data is binomial. When calculating Cohen's Kappa values, the linear model performed the worst, as expected with only a moderate level of agreement (k = 0.4371) (Lantz, 2015, pg. 323). The decision tree model performed the best (k = 0.820), indicating a very good agreement between predicted and actual values. The other models ranged from k-valued of 0.720 (reduced variable polynomial SVM) to 0.7712 (generic SVM, reduced variable SVM and the radial-basis kernel SVM). These models had a good agreement between the predicted and actual values. 

Removing redundant variables had little effect on the overall effectiveness of the model. In fact, it actually produced a slightly poorer performance in the polynomial model (F1 = 0.868). False negatives are more serious in this model as *H. hudsonius* has several threatened subspecies in Colorado. So  classifying an *H. hudsonius* as an *H. princeps* (a false negative) is more serious than classifying an *H. princeps* as an *H. hudsonius* (a false positive). The model's with the fewest number of misclassified meadow jumping mice were the polynomial SVM and the decision tree models with a single misclassified meadow jumping mouse. 

The models in this project had two major issues. The first was that the data was unbalanced, which created problems in classification. Overall, there were considerably fewer data points for western jumping mice, which likely contributed to the higher number of western jumping mice being being misclassified in most models. The other major issue was the small size of the data. In most wildlife biology studies, datasets greater than 30 samples are often considered to be large enough. However, in machine learning this can create issues as most models rely on higher amounts of data. In hindsight, testing the model on species with more data points may have produced better results. 

Several variables did have redundancies. Elevation was strongly correlated with both grasslands and coniferous forests. This makes sense because coniferous forests are found at higher elevations than grasslands. There was also a strong correlation between river distances and grasslands. This makes sense because the meadow jumping mouse is generally found in the wetlands and grasslands found within 340-meters of water (Trainor et. al. 2012). 

**Using K-fold Cross Validation**

K-fold cross validation is frequently used as a technique for dealing with unbalanced data (Boyle 2019). K-fold cross validation was developed from code by scikit-learn developers (2020) and was used on the two reduced models. However, these models were not able to produce higher values for accuracy and F1-values than their counterparts. The most promising of these models was the cross-validated polynomial SVM, which produced a mean accuracy of 0.878 and a mean F1-value of 0.870. Surprisingly the decision tree model with cross-fold validation did not outperform the polynomial SVM. It had an accuracy of 0.876 and an F-1 score of 0.865. 

#### Cross Validation for the SVM with reduced variables and a Polynomial Kernel 


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>mean accuracy</th>
      <th>F-1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K-fold SVM</td>
      <td>0.851</td>
      <td>0.843</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K-fold Polynomial SVM</td>
      <td>0.878</td>
      <td>0.870</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K-fold Decision Tree</td>
      <td>0.876</td>
      <td>0.865</td>
    </tr>
  </tbody>
</table>
</div>



**Possible Concerns**

One of the concerns with this model is that it might be good at identifying the meadow jumping mice simply because there are so many samples of them in the data. Nearly all of the models had issues with identifying the western jumping mice with 7-9 of them being misclassified as meadow jumping mice. In the future, this technique should be tested on species with more data points to see if the misclassification issues are real or simply a product of small sample sizes. In addition, techniques such as under-sampling the larger class could be used to try and make the data more balanced between samples. 

**Bibliography**

US Fish and Wildlife Service (January 6th, 2021) Preble’s Meadow Jumping Mouse Retrieved from: https://www.fws.gov/mountain-prairie/es/preblesMeadowJumpingMouse.php

Trainor, Anne M., Shenk, Tanya M. and Wilson, Kenneth R. (2012) Spatial, temporal, and biological factors associated with Prebles meadow jumping mouse (Zapus hudsonius preblei) home range. Journal of Mammalogy. 93(2), pgs. 429-438. Doi: 10.1644/11-MAMM-A-049.1

Python and R Tips (January 10, 2018) How to Create Pandas Dataframe from Multiple Lists? Pandas Tutorial. [Blog] Retrieved from: https://cmdlinetips.com/2018/01/how-to-create-pandas-dataframe-from-multiple-lists/

CMS WisCon(April 30, 2020) SVM Classifier in Python on Real Data Set [YouTube] Retrieved from: https://www.youtube.com/watch?v=Vv5U0kjYebM

Edpresso (2021) How to create a confusion matrix in Python using scikit-learn [Blog] Retrieved from: https://www.educative.io/edpresso/how-to-create-a-confusion-matrix-in-python-using-scikit-learn

Anita, Okoh (August 20th, 2019) Seaborn Heatmaps: 13 Ways to Customize Correlation Matrix Visualizations [Heartbeat] Retrieved from: https://heartbeat.fritz.ai/seaborn-heatmaps-13-ways-to-customize-correlation-matrix-visualizations-f1c49c816f07

scikit-learn developers (2007-2020) 3.1. Cross-validation: evaluating estimator performance. Retrieved from: https://scikit-learn.org/stable/modules/cross_validation.html

Boyle, Tara (February 3rd, 2019) *Dealing with Imbalanced Data: A guide to effectivly handling imbalanced datasets in Python*. [Towards Data Science] Retrieved from: https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18

Raschka, Sebastian and Vahid Mirjalili (2019) *Python Machine Learning Third Edition*. Packt Publishing Ltd.

Chen, Daniel y. (2018) *Pandas for Everyone: Python Data Analysis*. Pearson Education, Inc. 

Lantz, Brett () *Machine Learning with R: Second Edition*. Packt Publishing Ltd.

Ptonski, Piotr (June 22, 2020) Visualize a Decision Tree in 4 Ways with Scikit-Learn and Python [mljar] Retrieved from: https://mljar.com/blog/visualize-decision-tree/ 

DMNS Mammal Collection (Arctos). Denver Museum of Nature and Science. Accessed through Biodiversity Information Serving Our Nation (BISON) (n.d.) Zapus hudsonius prebli [Data File] Retrieved from https://bison.usgs.gov/#home on 1/16/2021

NatureServe Network Species Occurrence Data. Accessed through Biodiversity Information Serving Our Nation (BISON) (n.d.) Zapus hudsonius prebli [Data File] Retrieved from https://bison.usgs.gov/#home on 1/16/2021

Museum of Southwestern Biology. Accessed through Biodiversity Information Serving Our Nation (BISON) (n.d.) Zapus hudsonius prebli [Data File] Retrieved from https://bison.usgs.gov/#home on 1/16/2021

Fort Hayes Sternberg Museum of Natural History. Accessed through Biodiversity Information Serving Our Nation (BISON) (n.d.) Zapus hudsonius prebli [Data File] Retrieved from https://bison.usgs.gov/#home on 1/16/2021

University of Alaska Museum of the North. Accessed through Biodiversity Information Serving Our Nation (BISON) (n.d.) Zapus hudsonius prebli [Data File] Retrieved from https://bison.usgs.gov/#home on 1/16/2021 

iNaturalist.org.Accessed through Biodiversity Information Serving Our Nation (BISON) (n.d.) Zapus hudsonius prebli [Data File] Retrieved from https://bison.usgs.gov/#home on 1/16/2021

Angelo State Natural History Museum (ASNHC).Accessed through Biodiversity Information Serving Our Nation (BISON) (n.d.) Zapus hudsonius prebli [Data File] Retrieved from https://bison.usgs.gov/#home on 1/16/2021

Charles R. Conner Museum.Accessed through Biodiversity Information Serving Our Nation (BISON) (n.d.) Zapus hudsonius prebli [Data File] Retrieved from https://bison.usgs.gov/#home on 1/16/2021

University of Colorado Museum of Natural History.Accessed through Biodiversity Information Serving Our Nation (BISON) (n.d.) Zapus hudsonius prebli [Data File] Retrieved from https://bison.usgs.gov/#home on 1/16/2021

US Geological Survey. US Department of the Interior. (2014). National Land Cover Database 2011 (NLCD2011). [Raster]. Multi-Resolution Land Characteristics Consortium (MRLC). Retrieved from https://www.mrlc.gov/nlcd11_data.php on April 4th 2017. 

CDPHE_user_commuity, Colorado Department of Public Health and the Environment (2/19/2018) Colorado County Boundaries [Data File] Retrieved from: https://data-cdphe.opendata.arcgis.com/datasets/colorado-county-boundaries on 1/16/2021

U.S. Geological Survey, National Geospatial Program (06/15/2020) Retrieved from: NHD 20200615 for Colorado State or Territory Shapefile Model Version 2.2.1 [Data File] https://viewer.nationalmap.gov/basic/?basemap=b1&category=nhd&title=NHD%20View#/ on 1/26/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 1-1. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 1-2. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 1-3. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 1-4. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 1-5. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 1-6. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 1-7. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 2-1. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 2-2. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 2-3. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 2-4. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 2-5. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 2-6. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 2-7. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 3-1. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 3-2. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 3-3. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 3-4. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 3-5. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 3-6. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 3-7. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 4-1. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 4-2. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 4-3. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 4-4. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 4-5. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 4-6. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

ColoradoView/UV-B Monitoring and Research (n.d.) Colorado Digital Elevation Model files - 1 degree: Section 4-7. [Data File] Retrieved from: https://www.coloradoview.org/aerial-imagery/ on 2/06/2021

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Final_Experimental_Model_based_on_Mamun_2019_VGG16.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KupfiUtaxA1w"
      },
      "source": [
        "**Using Sequential Neural Networks and Transfer Learning to Assess Biodiversity at Karoo National Park, South Africa**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3e6HAYSrxNEP"
      },
      "source": [
        "**Introduction**\n",
        "\n",
        "Camera traps offer an unprecedented method to gain a large amount of data on wildlife in a noninvasive manner. Camera traps are remotely triggered cameras that use motion sensors to take photos when animals walk by. Camera traps also present a great opportunity for harnessing big data in wildlife biology (Ahumada et. al. 2019). However, a limitation of this type of data is the time needed to identify the animals in the images. This is where convolutional neural networks come in as these machine learning algorithms can quickly identify large numbers of images. \n",
        "\n",
        "However, animals do not pose for pictures, and it is possible that the angle of the animal in a photo can affect the ability of the algorithm to identify the animal in question. This includes situations in which the animal is seen from the front or from the rear rather than from the side. In these situations, the same animal can look very different. In addition, sexual dimorphism can be a factor in identifying an animal in a camera trap image. This algorithm will attempt to discover if sexual dimorphism and the position of the animal can affect the ability of a convolutional neural network to identify a species. In addition, it is possible that different keras applications can have greater success than others in correctly classifying the different categories of animal involved with camera trap images.\n",
        "\n",
        "The goal of this assignment is to determine if the position of the animal and the type of Keras architecture affects the ability of a CNN to identify an animal. Data will be obtained from three datasets, Snapshot Karoo: Season 1, Snapshot Camdeboo: Season 1 and Snapshot Kgalagadi: Season 1. These datasets are part of the Snapshot Safari program and are available through the Labeled Information Library of Alexandria (LILA BC) (n.d.). This lab will look at three categories of South African herbivore, the bull greater kudu (*Tragelaphus strepsiceros*), the cow eland (*Taurotragus oryx*) and mountain zebra (*Equus zebra*). Datasets will include a control dataset, in which all data for each category of antelope will be included in a single file, and an experimental dataset in which the data for each category is separated by front, rear and side views of the animal. The performance of both the control group and the experimental group will be assessed to determine the effectiveness of the algorithm.\n",
        "\n",
        "In addition, the performance of the CNN on the control and experimental groups will be tested using a sequential neural network with several different Keras applications. These will include the ResNet101 architecture, the MobileNet architecture and the VGG16 architecture. The accuracy of the algorithm in identifying individual categories, a confusion matrix will also be produced to see which categories the algorithm has difficulty identifying. \n",
        "\n",
        "This lab will utilize code created by Iftekher Mamum (2019). This code was chosen for its ability to quickly iterate through epochs, its higher accuracy and the ease at which it can create and display a confusion matrix. \n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 16
        },
        "id": "lDUuG7oRW75u",
        "outputId": "6f609d8f-347e-4b38-d27b-ea7f8a1b399f"
      },
      "source": [
        "from IPython.core.display import display, HTML\n",
        "display(HTML(\"<style>.container { width:95% !important; }</style>\"))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "<style>.container { width:95% !important; }</style>"
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "M9vmgl97Pv_u"
      },
      "source": [
        "#Import libraries\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras import backend, models, layers, optimizers, regularizers\n",
        "from tensorflow.keras.utils import to_categorical\n",
        "from tensorflow.keras.layers import BatchNormalization\n",
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "from sklearn.model_selection import train_test_split # This method ended up not being used\n",
        "from sklearn import metrics\n",
        "from sklearn.metrics import confusion_matrix\n",
        "from IPython.display import display # This library ended up not being used.\n",
        "from PIL import Image # This library was used to help view images\n",
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # This library was used for data augmentation\n",
        "import os, shutil # This libraries were used to create file connections\n",
        "import cv2\n",
        "np.random.seed(42)\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import itertools\n",
        "import keras \n",
        "from sklearn import metrics\n",
        "from sklearn.metrics import confusion_matrix\n",
        "from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img\n",
        "from keras.models import Sequential\n",
        "from keras import optimizers \n",
        "from keras.preprocessing import image\n",
        "from keras.layers import Dropout, Flatten, Dense\n",
        "from keras import applications\n",
        "from keras.utils.np_utils import to_categorical\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.image as mpimg\n",
        "%matplotlib inline\n",
        "import math\n",
        "import datetime\n",
        "import time\n",
        "import tensorflow as tf"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dEaSe2OmXCCu"
      },
      "source": [
        "backend.clear_session()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Hv-GnVGOXGky",
        "outputId": "9d6cd658-f684-4944-b69a-98fd1f844262"
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/gdrive')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Mounted at /gdrive\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GxGf46E3L3z3"
      },
      "source": [
        "**Methods:**\n",
        "\n",
        "The final dataset was not a balanced dataset because most pictures of the animals were of the side view rather than views from the front or the rear. Diagonal animals are hard to classify so an effort was made to use pictures with a clear enough difference in the position of the animal. The dataset ended up having 788 total images. The experimental dataset ended up with fewer images, with 773, and it not clear why this is. Unfortunately, this error was discovered very late in the project’s process and there was not enough time to fix it. In the control dataset, the cow eland had 277 training images, 69 validation images and 86 test images for 432 images total. The bull greater kudu had 257 training images, 64 validation images and 80 test images for 401 images total. The data for the mountain zebra 254 training images, 64 validation images and 80 testing images. The experimental dataset included separate categories for the side, front and rear views of the animal. The side view of the cow eland included 182 training images, 46 validation images and 57 test images for a total of 285 images. The front view of the cow eland had 57 training images, 14 validation images and 18 testing images for a total of 89 images. The rear view of the cow eland had 37 training images, 9 validation images and 12 testing images for a total of 58 images. The side view of the bull greater kudu had 182 training images, 46 validation images and 57 test images for 285 total images. The front view had 46 training images, 11 validation images and 14 testing images for 71 total images. The rear view had 26 training images, 8 validation images and 10 testing images for a total of 44 total images. The mountain zebra’s side view had 165 training images, 41 validation images and 51 testing images for 257 total images. The front view of the mountain zebra had 48 training images 13 validation images and 15 testing images with a total of 75 images. The mountain zebra’s rear view had 28 training images, 7 validation images and 9 testing images. The data was divided by a 64%/16%/20% train/validation/test split using methods from Shah (2017). \n",
        "\n",
        "The model created by Iftekher Mamun (2019) resizes the images to a size of 224 by 224 pixels, an image size chose was also to make the model compatible with the VGG16 architecture that would be used by its convolutional neural network. and converts images into a numpy array which is loaded into a bottleneck file for the training, validation and testing data to use. This numpy array is built from the number associated with each pixel in the photo. The data generators then prepare the data for the convolutional network and most importantly create the labels needed for the confusion matrix to work. This is the key difference between Mamun’s code and the code I originally created, as this was the code that finally allowed me to create the labels needed to build the confusion matrix. The matrix needs a set of easily accessible labels to compare the to the model’s predictions. The model does this by converting the numpy array used in the bottleneck file into a categorical variable that can be used for the labels. Basically, it is a set of dummy variables used to create labels for the machine learner.\n",
        "\n",
        "The model was originally designed to run with 7 epochs and a batch size of 50 (Mamun 2019). However, the number of epochs was increased to 30 in this experiment, and the batch size reduced to 10 as the datasets used were smaller than the animal-10 dataset from Kaggle. The architecture used for the project was the VGG16 algorithm.\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8o6w0AcKXJRn"
      },
      "source": [
        "base_dir = '/gdrive/My Drive/Regis/Karoo/Exp_Data_Slim/'"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NINbBUgGZ10o"
      },
      "source": [
        "# In the original code by Mamun et. al. (2019) this was the default height and width for images\n",
        "img_width, img_height = 224, 224\n",
        "\n",
        "# This code creates the bottleneck file to process the images to a size usable by the VGG116 algorithm\n",
        "top_model_weights_path = '/gdrive/My Drive/Regis/Karoo/bottleneck_fc_model_exp.h5/'\n",
        "\n",
        "# These are the links to the training, validation and testing datasets\n",
        "train_data_dir = '/gdrive/My Drive/Regis/Karoo/Exp_Data_Slim/Train'\n",
        "validation_data_dir = '/gdrive/My Drive/Regis/Karoo/Exp_Data_Slim/Val'\n",
        "test_data_dir = '/gdrive/My Drive/Regis/Karoo/Exp_Data_Slim/Test'\n",
        "\n",
        "# Epochs and Batch Size\n",
        "epochs = 30 \n",
        "batch_size = 10 "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EAWbU4_NcoCx",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a246d604-9c99-43cb-ab74-f7dc286ec5dc"
      },
      "source": [
        "vgg16 = applications.VGG16(include_top=False, weights='imagenet')\n",
        "\n",
        "datagen = ImageDataGenerator(rescale=1. / 255)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5\n",
            "58892288/58889256 [==============================] - 0s 0us/step\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tq6Ao2AWa-F_",
        "outputId": "3c95bd12-3631-4931-ae0c-c2b0dbcce63e"
      },
      "source": [
        "# This code generates the training data\n",
        "start = datetime.datetime.now()\n",
        "\n",
        "generator = datagen.flow_from_directory(\n",
        "    train_data_dir,\n",
        "    target_size=(img_width, img_height),\n",
        "    batch_size=batch_size,\n",
        "    class_mode=None,\n",
        "    shuffle=False)\n",
        "\n",
        "nb_train_samples = len(generator.filenames)\n",
        "num_classes = len(generator.class_indices)\n",
        "\n",
        "predict_size_train = int(math.ceil(nb_train_samples / batch_size))\n",
        "\n",
        "bottleneck_features_train = vgg16.predict_generator(generator,\n",
        "                                                    predict_size_train)\n",
        "\n",
        "np.save('bottleneck_features_train.npy', bottleneck_features_train)\n",
        "end = datetime.datetime.now()\n",
        "elapsed = end-start\n",
        "print ('Time: ', elapsed)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Found 773 images belonging to 9 classes.\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py:1905: UserWarning: `Model.predict_generator` is deprecated and will be removed in a future version. Please use `Model.predict`, which supports generators.\n",
            "  warnings.warn('`Model.predict_generator` is deprecated and '\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "Time:  0:07:38.266877\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aykk4fFCQvXs",
        "outputId": "bb45eb79-b513-4e4b-8bbe-a8bde7bf53f9"
      },
      "source": [
        "#This code generates the validation Data\n",
        "start = datetime.datetime.now()\n",
        "\n",
        "generator = datagen.flow_from_directory(\n",
        "    validation_data_dir,\n",
        "    target_size=(img_width, img_height),\n",
        "    batch_size=batch_size,\n",
        "    class_mode=None,\n",
        "    shuffle=False)\n",
        "\n",
        "nb_val_samples = len(generator.filenames)\n",
        "num_classes = len(generator.class_indices)\n",
        "\n",
        "predict_size_val = int(math.ceil(nb_val_samples / batch_size))\n",
        "\n",
        "bottleneck_features_val = vgg16.predict_generator(generator,\n",
        "                                                    predict_size_val)\n",
        "\n",
        "np.save('bottleneck_features_val.npy', bottleneck_features_val)\n",
        "end = datetime.datetime.now()\n",
        "elapsed = end-start\n",
        "print ('Time: ', elapsed)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Found 194 images belonging to 9 classes.\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py:1905: UserWarning: `Model.predict_generator` is deprecated and will be removed in a future version. Please use `Model.predict`, which supports generators.\n",
            "  warnings.warn('`Model.predict_generator` is deprecated and '\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "Time:  0:02:00.555726\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0Bb3Xnx-Rtuz",
        "outputId": "37090e65-2b26-4544-cf3c-676fa63db4f0"
      },
      "source": [
        "# This code generates the test data\n",
        "start = datetime.datetime.now()\n",
        "\n",
        "generator = datagen.flow_from_directory(\n",
        "    test_data_dir,\n",
        "    target_size=(img_width, img_height),\n",
        "    batch_size=batch_size,\n",
        "    class_mode=None,\n",
        "    shuffle=False)\n",
        "\n",
        "nb_test_samples = len(generator.filenames)\n",
        "num_classes = len(generator.class_indices)\n",
        "\n",
        "predict_size_test = int(math.ceil(nb_test_samples / batch_size))\n",
        "\n",
        "bottleneck_features_test = vgg16.predict_generator(generator,\n",
        "                                                    predict_size_test)\n",
        "\n",
        "np.save('bottleneck_features_test.npy', bottleneck_features_test)\n",
        "end = datetime.datetime.now()\n",
        "elapsed = end-start\n",
        "print ('Time: ', elapsed)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Found 243 images belonging to 9 classes.\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py:1905: UserWarning: `Model.predict_generator` is deprecated and will be removed in a future version. Please use `Model.predict`, which supports generators.\n",
            "  warnings.warn('`Model.predict_generator` is deprecated and '\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "Time:  0:02:25.704806\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f6g9PEc6fhhP",
        "outputId": "dd7c6010-81f9-44e4-fadb-a0f59ade0cf2"
      },
      "source": [
        "# This code generates the training data \n",
        "train_generator = datagen.flow_from_directory(\n",
        "    train_data_dir,\n",
        "    target_size = (img_width, img_height),\n",
        "    batch_size = batch_size,\n",
        "    class_mode = 'categorical',\n",
        "    shuffle=False)\n",
        "\n",
        "nb_train_samples = len(train_generator.filenames)\n",
        "num_classes = len(train_generator.class_indices)\n",
        "\n",
        "# This code will load the converted images for the training data from the bottleneck file\n",
        "train_data = np.load('bottleneck_features_train.npy')\n",
        "\n",
        "# This code generates the labels for the training data\n",
        "train_labels = train_generator.classes\n",
        "\n",
        "# This code converts the training labels into a categorical class\n",
        "train_labels = to_categorical(train_labels, num_classes=num_classes)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Found 773 images belonging to 9 classes.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RwjaILgCFGfA",
        "outputId": "adb49a76-348f-4637-a1e2-c12d0c520972"
      },
      "source": [
        "# This code generates the validation data\n",
        "validation_generator_top = datagen.flow_from_directory(\n",
        "    validation_data_dir,\n",
        "    target_size = (img_width, img_height),\n",
        "    batch_size = batch_size,\n",
        "    class_mode = 'categorical',\n",
        "    shuffle=False)\n",
        "\n",
        "nb_val_samples = len(validation_generator_top.filenames)\n",
        "num_classes = len(validation_generator_top.class_indices)\n",
        "\n",
        "# This code will load the converted images for the validation from the bottleneck file, create labels and convert them into a categorical class the confusion matrix can use\n",
        "val_data = np.load('bottleneck_features_val.npy')\n",
        "val_labels = validation_generator_top.classes\n",
        "val_labels = to_categorical(val_labels, num_classes=num_classes)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Found 194 images belonging to 9 classes.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jjQixOGPIlUK",
        "outputId": "d2c19d29-60a6-4343-933b-ab5f06a33b73"
      },
      "source": [
        "# This code generates the testing data \n",
        "test_generator = datagen.flow_from_directory(\n",
        "    test_data_dir,\n",
        "    target_size = (img_width, img_height),\n",
        "    batch_size = batch_size,\n",
        "    class_mode = 'categorical',\n",
        "    shuffle=False)\n",
        "\n",
        "nb_test_samples = len(test_generator.filenames)\n",
        "num_classes = len(test_generator.class_indices)\n",
        "\n",
        "test_data = np.load('bottleneck_features_test.npy')\n",
        "test_labels = test_generator.classes\n",
        "test_labels = to_categorical(test_labels, num_classes=num_classes)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Found 243 images belonging to 9 classes.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PpWGGsKs7Mya",
        "outputId": "97f05669-48d0-45f0-eaa4-af45e923d47b"
      },
      "source": [
        "categories = test_generator.class_indices\n",
        "print(categories)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "{'Eland': 0, 'Eland_Front': 1, 'Eland_Rear': 2, 'Kudu_Bull': 3, 'Kudu_Bull_Front': 4, 'Kudu_Bull_Rear': 5, 'Mountain_Zebra': 6, 'Mountain_Zebra_Front': 7, 'Mountain_Zebra_Rear': 8}\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RCfBSEQbxzQH"
      },
      "source": [
        "**Creating the Confusion Matrix**\n",
        "\n",
        "Like the model originally created for this project, Mamun (2019) used a CNN called a sequential neural network. This used transfer learning to incorporate the VGG16 architecture. The initial layers forming the top of the model was set to false to avoid incorporating it into the model, and the weights were set to ‘imagenet’ as per Mamun (2019). The model flattened the data and included three hidden layers. The first two use a with 100 nodes and the second with 50 nodes, and both use a LeakyRelu activation function. The model incorporates two dropout layers, one with a dropout of 0.5 and one with a dropout of 0.3. A softmax activation function was used on the final hidden layer as it is a classification-based model (Chollet 2018)."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NzwT2n0AMBu7",
        "outputId": "8df20da0-e1f4-470d-ab88-39df55b51a88"
      },
      "source": [
        "# Convolutional Neural Network \n",
        "start = datetime.datetime.now()\n",
        "model = Sequential()\n",
        "model.add (Flatten (input_shape=train_data.shape[1:]))\n",
        "model.add(Dense (100, activation=keras.layers.LeakyReLU(alpha=0.3)))\n",
        "model.add(Dropout(0.5))\n",
        "model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3)))\n",
        "model.add(Dropout (0.3))\n",
        "model.add(Dense(num_classes, activation='softmax'))\n",
        "\n",
        "model.compile(loss='categorical_crossentropy',\n",
        "              optimizer=optimizers.RMSprop(lr=1e-4),\n",
        "              metrics=['acc'])\n",
        "\n",
        "history = model.fit(train_data, train_labels,\n",
        "                    epochs=epochs,\n",
        "                    batch_size=batch_size,\n",
        "                    validation_data=(val_data, val_labels))\n",
        "\n",
        "model.save_weights(top_model_weights_path)\n",
        "\n",
        "(eval_loss, eval_accuracy) = model.evaluate(\n",
        "    val_data, val_labels, batch_size=batch_size, verbose=1)\n",
        "\n",
        "print(\"[INFO] accuracy: {:.2f}%\".format(eval_accuracy * 100))\n",
        "print(\"[INFO] Loss: {}\".format(eval_loss))\n",
        "end= datetime.datetime.now()\n",
        "elapsed= end-start\n",
        "print ('Time: ', elapsed)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/30\n",
            "78/78 [==============================] - 4s 35ms/step - loss: 2.1767 - acc: 0.2640 - val_loss: 1.5971 - val_acc: 0.4691\n",
            "Epoch 2/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 1.6439 - acc: 0.4609 - val_loss: 1.4951 - val_acc: 0.4897\n",
            "Epoch 3/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 1.5584 - acc: 0.4591 - val_loss: 1.3461 - val_acc: 0.5619\n",
            "Epoch 4/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 1.4123 - acc: 0.5429 - val_loss: 1.3591 - val_acc: 0.5309\n",
            "Epoch 5/30\n",
            "78/78 [==============================] - 2s 28ms/step - loss: 1.2255 - acc: 0.5755 - val_loss: 1.3440 - val_acc: 0.5979\n",
            "Epoch 6/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 1.2044 - acc: 0.6097 - val_loss: 1.3338 - val_acc: 0.5464\n",
            "Epoch 7/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.9590 - acc: 0.6772 - val_loss: 1.3140 - val_acc: 0.5722\n",
            "Epoch 8/30\n",
            "78/78 [==============================] - 2s 30ms/step - loss: 0.8433 - acc: 0.7347 - val_loss: 1.2113 - val_acc: 0.6031\n",
            "Epoch 9/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.9013 - acc: 0.7256 - val_loss: 1.2585 - val_acc: 0.6237\n",
            "Epoch 10/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.7988 - acc: 0.7476 - val_loss: 1.0762 - val_acc: 0.6753\n",
            "Epoch 11/30\n",
            "78/78 [==============================] - 2s 28ms/step - loss: 0.7556 - acc: 0.7322 - val_loss: 1.1212 - val_acc: 0.6649\n",
            "Epoch 12/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.7135 - acc: 0.7715 - val_loss: 1.0779 - val_acc: 0.6804\n",
            "Epoch 13/30\n",
            "78/78 [==============================] - 2s 30ms/step - loss: 0.6391 - acc: 0.7958 - val_loss: 1.1305 - val_acc: 0.6546\n",
            "Epoch 14/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.5928 - acc: 0.8107 - val_loss: 1.0571 - val_acc: 0.6701\n",
            "Epoch 15/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.6117 - acc: 0.7963 - val_loss: 1.0893 - val_acc: 0.6598\n",
            "Epoch 16/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.5292 - acc: 0.8426 - val_loss: 1.0652 - val_acc: 0.6753\n",
            "Epoch 17/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.5349 - acc: 0.8340 - val_loss: 1.0221 - val_acc: 0.7165\n",
            "Epoch 18/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.5208 - acc: 0.8403 - val_loss: 1.1162 - val_acc: 0.6856\n",
            "Epoch 19/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.4388 - acc: 0.8608 - val_loss: 1.0217 - val_acc: 0.6856\n",
            "Epoch 20/30\n",
            "78/78 [==============================] - 2s 30ms/step - loss: 0.4635 - acc: 0.8587 - val_loss: 1.0419 - val_acc: 0.7113\n",
            "Epoch 21/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.4413 - acc: 0.8823 - val_loss: 1.1191 - val_acc: 0.7113\n",
            "Epoch 22/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.3780 - acc: 0.8800 - val_loss: 1.0170 - val_acc: 0.7010\n",
            "Epoch 23/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.3797 - acc: 0.8791 - val_loss: 1.0995 - val_acc: 0.7062\n",
            "Epoch 24/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.2890 - acc: 0.9235 - val_loss: 1.1272 - val_acc: 0.6804\n",
            "Epoch 25/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.3000 - acc: 0.9051 - val_loss: 1.0702 - val_acc: 0.7320\n",
            "Epoch 26/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.2822 - acc: 0.9031 - val_loss: 1.0637 - val_acc: 0.7320\n",
            "Epoch 27/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.2890 - acc: 0.9104 - val_loss: 1.2240 - val_acc: 0.7010\n",
            "Epoch 28/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.2710 - acc: 0.9033 - val_loss: 1.0564 - val_acc: 0.7320\n",
            "Epoch 29/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.2330 - acc: 0.9395 - val_loss: 1.0857 - val_acc: 0.7474\n",
            "Epoch 30/30\n",
            "78/78 [==============================] - 2s 29ms/step - loss: 0.2806 - acc: 0.9054 - val_loss: 1.0803 - val_acc: 0.7113\n",
            "20/20 [==============================] - 0s 5ms/step - loss: 1.0803 - acc: 0.7113\n",
            "[INFO] accuracy: 71.13%\n",
            "[INFO] Loss: 1.0803152322769165\n",
            "Time:  0:01:10.055283\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iIcnr1ojWW-s"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 573
        },
        "id": "BYcrD3APLDAS",
        "outputId": "12242605-ce12-4472-afbb-bfad6613e3f8"
      },
      "source": [
        "# Creating graphs for the training and validation loss and accuracy\n",
        "acc = history.history['acc']\n",
        "val_acc = history.history['val_acc']\n",
        "loss = history.history['loss']\n",
        "val_loss = history.history['val_loss']\n",
        "epochs = range(len(acc))\n",
        "plt.plot(epochs, acc, 'r', label='Training acc')\n",
        "plt.plot(epochs, val_acc, 'b', label='Validation acc')\n",
        "plt.title('Training and validation accuracy')\n",
        "plt.ylabel('accuracy')\n",
        "plt.xlabel('epoch')\n",
        "plt.legend()\n",
        "plt.xlabel('epoch')\n",
        "plt.legend()\n",
        "plt.figure()\n",
        "plt.plot(epochs, loss, 'r', label='Training loss')\n",
        "plt.plot(epochs, val_loss, 'b', label = 'Validation loss')\n",
        "plt.title('Training and validation loss')\n",
        "plt.ylabel('loss')\n",
        "plt.xlabel('epoch')\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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tWuXzpw5U8uXL6+1atXSpk2b6vLly7V37976XqBTzbRp07RZs2baqFEjveaaa3Tv3r2Hjjd06FBt3ry5NmrUSJdmcO/1ihUr9IwzztDmzZtr8+bNdebMmYfee/TRR7VRo0bapEkTvfPOO1VV9ZdfftFzzjlHmzRpos2bN9fly5cftc9of04uj2bOVK1e3W7VBNWyZVUbN7YOVgMHqg4frvr++6pz56pu3Gj3+LsCg2xuLS10JYNojGB9/PHH07JlS6ZMmUKXLl0YN24cV1xxBSLCww8/zPHHH8/Bgwc555xz+PHHH2nSpEmG+5k3bx7jxo1jwYIFpKSk0KJFCxITEwHo1q0b1113HQD33HMPr7zyCgMHDqRz58506tSJyy677Ih97d27lz59+jB9+nTq1avH1VdfzXPPPcett94KQMWKFZk/fz6jR49m+PDhvPzyy0ds70Ndu0NSU+Gxx2x8npo1rWTQogWUK+ff5AuRaPczKDTSqorAqojS5hN49913adGiBc2bN2fx4sUsWbIk03188803dO3alVKlSnHsscfSuXPnQ+/973//4+9//zuNGzfmrbfeYvHixVnGs2zZMmrXrk29wJgovXv3ZsaMGYfe79atGwCJiYmHBrcLduDAAa677joaN27M5ZdffijuUIe6TnvfxbgNG2zcnbvusrF65s+Hc86x6h9PBIVKoSsZRGsE6y5dunDbbbcxf/58du/eTWJiIitWrGD48OHMnTuX8uXL06dPH/bu3Zur/ffp04cJEybQtGlTXnvtNb766qs8xZs2DHZmQ2D7UNeOadNspM1t22xkzr59PQEUYl4yCJMyZcpw1llnce211x4qFWzfvp3SpUtz3HHHsWHDBqZMmZLlPs4880wmTJjAnj172LFjB5988smh93bs2MGJJ57IgQMHeOuttw4tL1u2LDt27DhqX6eeeiorV65k+fLlgI0+2q5du5DPx4e6jmMpKXD33TaRSoUKMHcuXHedJ4JCzpNBGPXs2ZOFCxceSgZNmzalefPm1K9fn169etG2bdsst2/RogXdu3enadOmXHDBBZx++umH3nvwwQdp1aoVbdu2pX79+oeW9+jRgyeeeILmzZsfMT9xyZIlGTNmDJdffjmNGzemSJEi9OvXL+Rz8aGu49Tvv0P79vDII3bf/9y50KhRtKNy+UCskTl2JCUlaXJy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wnPMrAweDCVLWiN3xGSVKfLywBLAr0AdoDiwEGiYxfoDgVez26+XDMLs00/tK8ewYaFvs3On6ksvqT78sOquXZGLrZB74gn71T/55JHLP/9c9ZhjVBs2VF23LjLH3rZN9eKL7fh9+1oBMCdSU6166667VOvVs/2IqJ55pupTT6muWXN43ZQU1eXLrSD56KOqV1+tmpRkzVPBJY4778x5c9O6darHHqvaoUPeqlAyk5pqtaWges01qh98YHX4SUlWqoslRKuaCGgNTA16PQQYksX63wHnZrdfTwZhtH27avXqVumZ06uBy5MZM1QTEheCvl0AAB4USURBVFQvvTTji9iXX1rVzamnqq5dG95j//KLJZqEBNWnn877RTQ1VXXRItX77rM/pbSL++mnqzZtavXqwRf9k06yi/fNN6s+/7zq11+r3nCDvXfRRapbt4Z+7F69rGpt2bK8nUNWUlPt3NLib9UqNhv6o5kMLsOqhtJeXwU8k8m6NYF1QEIm718PJAPJNWrUiNgvK+4MHGhf5777LtqRxJX1661+u27drOu4Z8ywb89166quXh2eY3/+uWr58tZENH16ePaZ3tKlqg89pNqmjeoFF9hdOa+8Ys1KWV1ER4+29oj69VV//jn740yfblewe+8NX+xZGTFC9fLLw9cukd9iJRncCTwdyn69ZBAm331niWDgwGhHEldSUlTPPtu+LS9cmP36M2daG32dOqorV+b+uKmpqqNGWWngtNNUf/019/uKpK++Uq1QQbVcOdWpUzNfb+9eKzXVqaO6e3f+xRfLsksGkex0thaoHvS6WmBZRnoA70QwFhds/35r/atWDR5+ONrRxJVhw2wit9GjoUmT7Ndv08YakjdtsiEKcjMj1r591pnq1lutR+usWTZuYEHUrp3NaVS9ug329uSTGXf+GjECli2zOQKOOSb/4yyUssoUevib+4fARUCRUNYPbFMU+A2ozeEG5NMyWK8+sBKQUPbrJYMwuP9+KxROmhTtSOLK5Mn2a7/22pxvO2+eVe9Ur251/qFat87uGE6rTkl/i2RBtWOHarduFnefPkd2uPrtNytZXXpp9OKLRWRTMhDNKO2mIyIdgGuAvwHvAWNUdVkI210IjMLuLHpVVR8WkQcCQU0MrDMMKKmqg0NJXklJSZqcnBzKqi4jS5ZAs2Zw+eU2kIvLF7//Ds2b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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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0EPp2LEjP/74IykpKVx11VVMnDiR+vXrc/755+cZ37Bhw2jVqhXjx49nwoQJXHLJJUyfPp2RI0fy1FNP0bFjRzZv3kyFChUYNWoU3bt358477yQjI4OtW7fm+/dRWF7PAnDggZYgPo9OM4hzrmidffbZJCQkALBhwwbOPvtsjj76aIYMGcKsWbOy3ea0006jfPny1KpViwMPPJC///57v3XatWtHnTp1KFOmDC1btmTx4sXMmTOHBg0a7OmfEE6i+OGHH7j44osBOPHEE1mzZg0bN26kY8eODB06lMcff5z169eTmJhI27Ztefnllxk+fDgzZswgKSmpoL+WAvMSRabu3eGhh2DDBjjggFhH41z8KcA3/2ipXLnynud33303Xbt25b333mPx4sV06dIl223Kly+/53lCQgLp6ekFWqcwbrvtNk477TQ++eQTOnbsyOeff06nTp2YOHEiH3/8Mf3792fo0KFccsklET1uXrxEkal7d8jIgAkTYh2Jcy6CNmzYQO3a1l/3lSjc3di4cWMWLlzI4sWLARg7dmye25xwwgmMHj0asLaPWrVqUbVqVRYsWEDz5s259dZbadu2LXPmzGHJkiUcdNBBXHnllVxxxRVMmzYt4ueQF08UmY49FpKSvPrJuRLmlltu4fbbb6dVq1YRLwEAVKxYkaeffpoePXrQpk0bkpKSOCCPWonhw4czdepUWrRowW233carr74KwKOPPsrRRx9NixYtKFu2LD179uTbb7/lmGOOoVWrVowdO5brr99vfNWoK1FzZqekpGihJi4680y782nRIvDeps7l6Y8//qBp06axDiPmNm/eTJUqVVBVBg8eTKNGjRgyZEisw9pPdp+XiExV1VzvE/YSRagePWDJEvjzz1hH4pyLI88//zwtW7bkqKOOYsOGDVx11VWxDimivDE7VPfu9vOzz6Bx49jG4pyLG0OGDCmWJYhI8RJFqHr1LEF4O4Vzzu3hiSKr7t3h229tsEDnnHOeKPbTowds2wYx6CbvnHPFkSeKrDp3hvLlrZ3COeecJ4r9VKoEnTp5O4VzcaBr1658nuV/9dFHH2XQoEE5btOlSxcyb6M/9dRTWb9+/X7rDB8+nJEjR+Z67PHjxzN79t6Zne+55x6+isC8NsVxOHJPFNnp3h1mzYKlS/Ne1zkXM+effz5jxozZZ9mYMWPCGm8JbNTXatWqFejYWRPFiBEjOOmkkwq0r+LOE0V2Mm+T/eKL2MbhnMtVv379+Pjjj/dMUrR48WKWL1/OCSecwKBBg0hJSeGoo45i2LBh2W5fr149Vq9eDcADDzzAkUceyfHHH79nKHKwPhJt27blmGOOoW/fvmzdupVJkybxwQcfcPPNN9OyZUsWLFhA//79GTduHABff/01rVq1onnz5gwYMIAdO3bsOd6wYcNo3bo1zZs3Z86cObmeX3EZjtz7UWTnqKOgdm1rp7j88lhH41xciMUo4zVq1KBdu3Z8+umn9OnThzFjxnDOOecgIjzwwAPUqFGDjIwMunXrxu+//06LFi2y3c/UqVMZM2YM06dPJz09ndatW9MmmHLgrLPO4sorrwTgrrvu4sUXX+Taa6+ld+/e9OrVi379+u2zr+3bt9O/f3++/vprjjzySC655BKeeeYZbrjhBgBq1arFtGnTePrppxk5ciQvvPBCjudXXIYj9xJFdkSsVPHVVxCFsWGcc5ETWv0UWu309ttv07p1a1q1asWsWbP2qSbK6vvvv+fMM8+kUqVKVK1ald69e+95b+bMmZxwwgk0b96c0aNH5zhMeaa5c+dSv359jjzySAAuvfRSJk6cuOf9s846C4A2bdrsGUgwJ8VlOHIvUeSkRw946SX45ReI0Jy5zpVksRplvE+fPgwZMoRp06axdetW2rRpw6JFixg5ciRTpkyhevXq9O/fn+0F7BvVv39/xo8fzzHHHMMrr7zCt99+W6h4M4cqL8ww5UU9HLmXKHJy0kk2f7bf/eRcsValShW6du3KgAED9pQmNm7cSOXKlTnggAP4+++/+fTTT3PdR6dOnRg/fjzbtm1j06ZNfPjhh3ve27RpE4cccgi7du3aMzQ4QFJSEps2bdpvX40bN2bx4sXMnz8fgNdff53OnTsX6NyKy3DknihyUr06tG/v/SmciwPnn38+v/32255EkTksd5MmTbjgggvo2LFjrtu3bt2ac889l2OOOYaePXvStm3bPe/dd999tG/fno4dO9KkSZM9y8877zwefvhhWrVqxYIFC/Ysr1ChAi+//DJnn302zZs3p0yZMlx99dUFOq/iMhx51IYZF5GXgF7ASlU9Opv3bwYuDF4mAk2BZFVdKyKLgU1ABpCe1xC4mQo9zHhW995rj1WroGbNyO3XuRLChxmPL8VxmPFXgB45vamqD6tqS1VtCdwOfKeqa0NW6Rq8H1aSiIoePUAVvvwyZiE451ysRS1RqOpEYG2eK5rzgbeiFUuBpaRAjRreTuGcK9Vi3kYhIpWwksc7IYsV+EJEporIwDy2HygiqSKSumrVqsgGl5Bgjdqff24lC+fcfkrSLJklWWE+p5gnCuB04Mcs1U7Hq2proCcwWEQ65bSxqo5S1RRVTUlOTo58dD16wIoVMGNG5PftXJyrUKECa9as8WRRzKkqa9asoUKFCgXavjj0oziPLNVOqros+LlSRN4D2gETs9k2+k45xX5+9hnk0KvTudKqTp06pKWlEfHSvIu4ChUqUKdOnQJtG9NEISIHAJ2Bi0KWVQbKqOqm4PkpwIgYhWhDeTRvbtVPt9wSszCcK47Kli1L/fr1Yx2Gi7KoJQoReQvoAtQSkTRgGFAWQFWfDVY7E/hCVbeEbHoQ8J6IZMb3pqrGtjND9+7w2GOweTNUqRLTUJxzrqhFrR9FLES8H0Wmr7+2Ru0PP4RiNk68c84VRqz7UZQcxx9vExp5L23nXCnkiSIc5ctD167en8I5Vyp5oghX9+4wfz6EjOninHOlgSeKcPUIRiPxUoVzrpTxRBGuhg2hfn1PFM65UscTRbgyZ737+mvYsCHW0TjnXJHxRJEfV14JW7fCPffEOhLnnCsynijyo3VrGDQInnwSfv011tE451yR8ESRX/ffD7VqWcLYvTvW0TjnXNR5osiv6tVh5Ej4+Wd48cVYR+Occ1HniQJ46y343//yscFFF0GnTnDbbbB6ddTics654qDUJ4o1a+Af/7AbmtasCXMjEXj6adi40ZKFc86VYKU+UdSsCePHw6JFcPrpdlNTWI46CoYMseqnSZOiGqNzzsVSqU8UAJ07w+jRMHkynHcepKeHueE990CdOtawHfZGzjkXXzxRBPr2tbteP/zQrvthjb5epYrNU/H777axc86VQJ4oQvzjH3DXXfDCCzBsWJgbnXkm9OxppYvly6Man3POxYIniixGjIABA+C+++CZZ8LYQASeeAJ27oQbb4x6fM45V9Q8UWQhAs89ZxPZDR4M774bxkZHHAG33w5jxsBXX0U9RuecK0qeKLKRmAhjx0L79nDBBTBxYhgb3XqrJYzBg2HHjqjH6JxzRSVqiUJEXhKRlSIyM4f3u4jIBhGZHjzuCXmvh4jMFZH5IhKTjgqVKsFHH0G9etC7N8yYkccGFSpYg/aff8IjjxRFiM45VySiWaJ4BeiRxzrfq2rL4DECQEQSgKeAnkAz4HwRaRbFOHNUs6ZNP1GpkrVX59l7u0cP6NfPGjgWLSqSGJ1zLtqilihUdSKwtgCbtgPmq+pCVd0JjAH6RDS4fDj8cPjsM9i0yfLA2rzO6D//gYQEuP76IonPOeeiLdZtFMeKyG8i8qmIHBUsqw0sDVknLViWLREZKCKpIpK6atWqqATZogW8/75Nl927dx596+rUgeHDrUPG++9HJR7nnCtKsUwU04DDVfUY4AlgfEF2oqqjVDVFVVOSk5MjGmCoLl3gpZfgxx/h8cfzWPn66+Hoo+Haa2Hz5qjF5JxzRSFmiUJVN6rq5uD5J0BZEakFLAPqhqxaJ1gWcxdcAKedZn3rli7NZcWyZe0e26VLrXThnHNxLGaJQkQOFhEJnrcLYlkDTAEaiUh9ESkHnAd8EKs4Q2X2rdu9G264IY+VjzvOpk599FGYPr1I4nPOuWiI5u2xbwE/AY1FJE1ELheRq0Xk6mCVfsBMEfkNeBw4T006cA3wOfAH8LaqzopWnPlVvz7cfbd1xPv44zxWfvBBu3XqqqsgI6NI4nPOuUgTDWv0u/iQkpKiqampUT/Ozp3QsiVs2wazZtntszkaPdomOnrqKRtMyjnnihERmaqqKbmtE+u7nuJSuXLw7LOweLFNoZ2rCy6Ak06yIT5WrCiK8JxzLqI8URRQp07Qvz88/LCVKnKUORvejh1hNGw451zx44miEP71L0hKshqlXGvwGjWCO++Et9+23nvOORdHPFEUQnKyJYuJE+G11/JY+ZZboHFjyyphz7fqnHOx54mikAYMsDthb7oJ1qzJZcXy5a1hY9GiMBo2nHOu+PBEUUhlytgER+vWwW15jXPbpQtceqk1bMzMdlBd55wrdjxRRECLFjBkiE2hOmlSHiuPHAkHHABXX20995xzrpjzRBEhw4ZB3bp2/d+1K5cVa9WyEsWPP9rgUc45V8x5ooiQKlVseI8ZM+Cxx/JYuX9/u7/2lltg5cqiCM855wrME0UE9ekDp59u4wDmOsmRiDVsb97MpuvuJC3Na6Gcc8VXYqwDKGmeeAKaNbORxt97z5Zt2ADz5+/7mDevKfPLreHvsUkwFipXhqZNbdvMn82a2dhSCQmxPSfnXOnmiSLCDj/c2ituvRVSUqxkkXU+pUMPhYYN4bS+FWn00cNUTdjMn2ffxex5Zfn66337ZJQvD02a7Js46ta1+ZFq17b3nXMumjxRRMGQIfDTT7B+PZxxhiWFzMcRR1jpwSTChDbQrRuwGr54CrASyB9/wOzZex8//QRvvbX/sQ480JJGZvLI/Fm/PrRqBRUrFtVZO+dKKh89tji48Ub497/ho49sZqQcbNlicyEtXQppafbI+nz9+r3rJybarbsdOkD79vazUSNrInHOOQhv9FhPFMXB9u3Qrh38/bfdNnXggQXe1ebNsGwZzJ0LP/9sj19+gU2b7P3q1S1phD5q1IjQeTjn4o4ningyYwa0bQunnALvvx/Rr/0ZGTBnDkyebIlj8mQb8TbzTqs77/RRRZwrrXw+injSvLnNiPfhhzBqVER3nZAARx0Fl19uu/79d6uimjDBpst44IEw+n4450otb8wuTq67zuZXHTLExoVq3Dhqh0pKgq5drd/ftm12yEMOgXPOidohnXNxyksUxUmZMvDqq3ar0kUX5TEWSGQkJNhsrccdBxdfDN98E/VDOufijCeK4ubQQ61+KDUV7r23SA5ZsSJ88IHdunvGGVY15ZxzmaKWKETkJRFZKSLZjqctIheKyO8iMkNEJonIMSHvLQ6WTxeROG2dLoS+feGyy+Cf/4QffiiSQ9aoYZPvJSVBz555DEHinCtVolmieAXokcv7i4DOqtocuA/I2oLbVVVb5tUaX2I99hjUq2dVUBs2FMkhDzsMPv3U+mv06AFr1+Z/H0uW2Lwcv/4a+ficc7ERtUShqhOBHC81qjpJVdcFLycDdaIVS1xKSoLXX7dedNdeW2SHbd4cxo+HBQtsgMNt28LbbvlyGDzYOvQ99BAMHJjHPOLOubhRXNooLgc+DXmtwBciMlVEBua2oYgMFJFUEUldlXVQpXh33HFw112WMMaOLbLDdukCb7xhw4ZccIH1w8jJypUwdKi1b4waZVPD3n+/NbF8/HGRheyciyZVjdoDqAfMzGOdrsAfQM2QZbWDnwcCvwGdwjlemzZttMTZuVO1XTvVatVUly4t0kM/9pgqqF59teru3fu+t3q16m23qVaqpFqmjGr//qoLFuwNuUED1TZt9t/OOVe8AKmax7U1piUKEWkBvAD0UdU1mctVdVnwcyXwHtAuNhEWA2XL2tf7Xbtsvu0inLjiuutsbqVnn7VOeWDNJcOG2aCDDz1kc3DMng0vvwwNGuwN+a67YOpU6z/oXEm2Y4f9n8yeHetIoiivTFKYB7mUKIDDgPnAcVmWVwaSQp5PAnqEc7wSWaLI9Pzze7/eb9tWZIfNyFC96CI79CWXqFavbs/POkt1xoyct9u1S/WII1RbtfJShSvZ/vUv+5847rj4/FsnjBJFNJPEW8AKYBeQhrVDXA1cHbz/ArAOmB48UoPlDbDqpt+AWcCd4R6zRCeK3btVhw61j+yYY1RnzSqyQ+/YoXrKKXboXr1Up00Lb7tXXrFt3nsvuvE5Fyt//aWalKR68MH2t/7OO7GOKP/CSRQ+KGC8+egj62OxZQv85z92e1ERjBu+Y4fd+nrkkeFvk55us/VVqQLTpvnw5q7kGTjQql1/+w3OPttqiGfNsurXeOGDApZEvXpZ1+njj4err4azzoI1a/LerpDKl89fkgCbD+Puu2H6dLvl1rmSZPp0eOEFuOYam33yoYdg3jx4/vlYRxZ5XqKIV7t3W4ni9tshOdkavLt2jXVU+0lPt3+iihWtE14Z/2riSgBVOPFEmx1g3jyb50XV/gVnz7Z+SElJsY4yPF6iKMnKlLGZ8SZPtrqdbt3gjjuKZCDB/EhMhHvusULQe+/FOhrnIuO99+Dbb2HECEsSYFWrDz8Mq1bZz5LESxQlwZYtcMMNVg5u2xbefNMm6C4mMjJsPoyyZa0u10sVLp7t2GFtb5UrWyk5MctkDeefb4NszptnY3wWdxErUYjI9SJSVcyLIjJNRE6JTJiu0CpXtorR//7X/jpbtbKqqGIiIcFKFTNnwjvvxDoa5wrn0Udh0SKb5j5rkgDrc7Rrl/U3KinC/W43QFU3AqcA1YGLgQejFpUrmH79rI6nVSubXOLHH2Md0R7nnmvfwu69t0j7DDoXUX/9ZYng9NPh5JOzX6dBAxv37KWX7A6okiDcRJF5Y+OpwOuqOitkmStO6taFTz6xoWCvugp27ox1RMDeUsWsWVbwKQ5KUK2rKyJ33QXbt8Mjj+S9XlKSjaRcEoSbKKaKyBdYovhcRJIA/15YXFWpAk89ZVflkSNjHc0eZ59td0Dde2/uAw1GW3o6PP201R9fdZXVObv4pGp/5g8+aH9X0RyR/9dfrZRw7bU2SnJuata0GxI/+sgaveNeXj3ygsbuMkBroFrwugbQIpxti/JRontmF0TfvqoVKqjOmxfrSPYYO9Z6sL71Vnjrb92q+vjjqk2b2rAhv/xSuON//rlqs2YWQ/Pm9rN9e9W0tMLttzhYuVJ10qTYHHv1ahtd5ocfon+snTtVv/pK9frrbfBJSxeqIqqHHKL69tuRH0pj927VTp1Ua9VSXbcuvG22blWtW1c1JcWGwimuiNQQHkBHoHLw/CLg38Dh4WxblA9PFFksW2bjC5x0UrEZhCYjQ/Xoo1WbNFFNT895vc2bVR95ZO/QCG3b2gC6oNqtm10o8nNKc+fa8CNgF7MguCMAAB4wSURBVJd337Xt33lHtXJlO05RXOSiJS3NxtYC1RdeKNpj//HH3mNXrar622+RP8aaNapvvKF67rmqBxxgxypfXvXUU1WfecYGVp4yRbV1a3uvZ0/VhQsjd/z//tf2+8wz+dvu1Vfz98UoFiKZKH7H2iSOAX4FBgPfhbNtUT48UWTjySftY37jjVhHskfmP93o0fu/t3Gj6oMPqiYn2zonnqj67bf23oYNNgBbZvJo184u+Ll9W1u3TnXIENXERMuZDz2kun37vuvMnKnasKFq2bJ2ISgmOTVsK1aoNm5s53f88fbN+rXXiubYX3xhF+4DD1QdN061dm37Vr94ceH3nZGh+tRTqp07qyYk2Gd+0EGql1+uOn68fZnIatcu1UcfVa1SRbViRdV//tNKIIWxbZtqvXpWAt21K3/bpqfb0Gz16+//d1dcRDJRTAt+3gNcHrqsOD08UWQjPd3qVpKT7WtZMZCRYf90jRvvLVWsX696332qNWrYX2X37jl/w9+2TfXZZ/dWOzRtagMQhl4Qdu2yi36tWnbhvOIKG8AtJ+vW2bdTsHUL8k+dnm5VP4sW5X/bglq50qrSKldW/f57q+448USbI2Ts2Oge+6mn7ALevPnexDBzppX8Gje26qiC2r7dSg+ZVYR33qk6eXL4VThLl1pVJagedVThSov/93+2n6++Ktj2n39u2//nPwWPIZoimSi+A24H5gEHB20WM8LZtigfnihyMH26/UdffnmsI9lj3Dj763viCdV77tlbndCrl+rPP4e3j127VN98U7VFC9v2sMOsPePTT616C6xeOdzRbtPT7YKUn3aL3bvtAnbddfZtN7O+/IgjVAcOtIv1qlXhHT+/Vq+2c69YUfWbb/Yu37xZ9YQT7COPxsi9u3apXnvt3s9r48Z935840aqFOnRQ3bIl//vfsMGSHVgJsjAlvA8/tL+LzC8A+f2utHy5lU769Cl4DKqqJ59sX4Lyat9IT7ff3w03WPVs48ZWcj75ZNV+/exfeOhQ1XvvtZLTyy9bqfq77woeWyQTxcHAUOCE4PVhwCXhbFuUD08Uubj5Zvu4C/MXFUEZGXsv8KB6xhmqU6cWbF+7d6t+9JFqx45791evniWjglxkMtstDjrIvqVnZ/Zs1bvu2luqKV/evsG+9ZbNDNi7t9XXZ8bTsqXqjTeqfvKJ6qZNBTvPUOvWWX18+fJW/ZPVxo12oS5bVvXjjwt/vEzr11tpD+x8cmpnevddK9X06pW/6prly+13lZgYueqzzZtVb7rJEmdysu133brwSieXXWa/wz//LFwMv/5qJdtbbtn/vR077MvNlVdaFV7m31PPnqrnnGO/7w4drOR86KH2t5n5d5X5OOiggscWsURh++IgoFfwODDc7Yry4YkiF5s329WzSZNiU1n600+qV10V2cbP77+3aqjCzu2U2W6RmKj69NOWcP73P/uG27Kl/eeUKWPf9F5+2S6gWe3aZed4//2qXbuqlitn25Uta9/4H3ywYLWBGzbYt8xy5Szx5GTdOpuONqdkkl8LFlg1V2Ki6qhRea//zDN2vpdfHl7CnjvX/kQrV7YLZ6RNn24X3MyLa5kyqjVrqh55pC0/9VSbpOu66+wb+wMP2MX9ppsic/xLLrHPYskS+3d85x3VCy/cW5quUsWq28aO3b+UltWuXfa3s3ChJaHJkwseVyRLFOcAS4BXgdeARUC/cLYtyocnijx88ol95CNGxDqSuBDabtG06d4LTPv2VmpYsSJ/+9uyxS7Yt95qF3Cwi+LQoeFPh75pk5WcEhNV338/7/VDq6cybwooiIkTrb2nenXVCRPC3+7uu+0877479/V+/tn2n5xc+Fugc5ORYdVx//63lQgHDbKL88kn22dSv/7eCzdYw3x2XwIKYskSSxQNGtjnAZaoBgywKrIinLhyH5FMFL+FliKAZOC3cLYtyocnijCcc479tc6dG+tI4kJ6uurw4faN8777VOfPj9y+f//dvsEmJFgpY8AA1Tlzcl5/y5a9dwCNGxf+cf7+e2+D948/5j/OV16x+I48Mv9VMLt3W4kCrGSWnU8+Ua1UyS7Sha3iiZRdu6xtKa9v9vn1wANWarrmGku4+b2LKhoimShmZHntjdnxavly+8rUtWv83QdaQi1apDp4sPWNFMm+Y+G2bdYdpkwZa8DPr+XLVRs1snaTvL6x//WX3X56661WRZZ5m/Latfk/rqpdDHv1snPLOlXoK69Y4mvVKv8lNBcZkUwUDwOfA/2Dx6fAQ+FsW5QPTxRhyqw8fvXVWEfiQvz9t911lVn10a2b6pdfWpNSz552oX3llYLvf+lS+9ZerZrVa6vaLcVTptjdYhdcYO9nVrskJlpHx2HDCt8XYcsWK5WVL2/VWLt3WxtN5nlu2FC4/buCCydRhD0fhYj0DXpoA3yvqsVuGppSOx9Ffu3ebVOp/vknzJkDtWrFOiIXYuNGeO45G8b6r7/s41m92kaSv+KKwu178WLo3NmmMGnaFFJTbZA7sLGvjj0WOnSwn61b28yEkbJmDXTsCH//Db17w2uv2dwNr7wC5cpF7jguf8KZjyKqExeJyEvYXVIrVfXobN4X4DFssMGtQH9VnRa8dylwV7Dq/ar6al7H80SRDzNn2nDkF11ks8O7Ymf7dnj9dUsaAwfaIxIWLIDzzrO5FDKTwrHHQp06NktbNC1ZYsdascLm2nrkEZ/IKtYKnShEZBOQ3QoCqKpWzSOATsBm4LUcEsWpwLVYomgPPKaq7UWkBpAKpATHnwq0UdV1uR3PE0U+3XEH/POf8OWXcNJJsY7GlRILFti0KWecEf3E5PJW6BnuVDVJVatm80jKK0kE208E1uaySh8siaiqTgaqicghQHfgS1VdGySHL4EeeR3P5dPdd0OTJjar0Jw5sY7GlRJHHAFnnulJIp7EutBXG1ga8jotWJbT8v2IyEARSRWR1FWrVkUt0BKpYkWb5CgxEXr2tApx55zLItaJotBUdZSqpqhqSnJycqzDiT/168PHH8PKlXDaabB5c6wjcs4VM7FOFMuAuiGv6wTLclruoiElxeYn/e03OOccmwLOOecCsU4UHwCXiOkAbFDVFVifjVNEpLqIVAdOCZa5aDn1VHjmGfj0Uxg0yCeUds7tkRjNnYvIW0AXoJaIpAHDgLIAqvos8Al2x9N87PbYy4L31orIfcCUYFcjVDW3RnEXCVdeCUuXwn33wWGHWWO3c67Ui2o/iqLmt8dGgCpcdhm8+qr1r+jfP9YROeeiqNC3x7pSSMS6AJ98spUwvvgif9svXw7PPguLFkUnPudckfNE4fZXtiyMGwdHHQV9+8L06bmvv2mTjcdwyilQt661cXTrBn67snMlgicKl72qVa2PRfXq1tC9ZMm+76enW8P3hRfCQQfBpZfC/Plw552WZP76y7reZg4k5JyLW1FtzHZx7tBDLRkcf7x1yPvxR0sGb7wBY8ZY34saNawd46KLbBCf0O62/fpZe8fo0T6gj3NxzBOFy91RR8H48XurlbZsgfLl4fTTLTn07Jn90J99+8KDD8Jtt0HDhnYnlXMuLnmicHnr3NlKEM8/D2edZSWFatXy3u6WW2DePLj/fksWl14a/VidcxHnt8e66Nq1y0odEyfaHVRdusQ6IudcCL891sVe5h1UDRtaaWTu3FhH5JzLJ08ULvqqVbOBBxMTbeDB1atjHZFzLh88UbiiUb8+vP8+pKXZZAQ7dsQ6IudcmDxRuKJz7LHWMe+HH2DAAB940Lk44Xc9uaJ1zjl7O+Y1agTDh8c6IudcHjxRuKJ3++3w559w773WyH3RRbGOyDmXC696ckVPBEaNsltlL7kErr0WNm6MdVTOuRx4onCxUa4cfPABDB4MTz0FTZvCO+94u4VzxZAnChc7SUnwxBMweTIkJ1uP79699x+A0DkXU54oXOy1awepqfDIIzBhAjRrZs997m7nigVPFK54SEyEoUNh9mw48US46SZo2xamTMl7W+dcVHmicMXL4Ydb28U779gw5u3bw3XXeWO3czEU1UQhIj1EZK6IzBeR27J5/z8iMj14/Cki60Peywh574NoxumKGREbF+qPP+Caa+DJJ62x+803YffuWEfnXKkTtUQhIgnAU0BPoBlwvog0C11HVYeoaktVbQk8Abwb8va2zPdUtXe04nTFWNWq8Pjj1th98ME2m1779vDdd7GOzLlSJZolinbAfFVdqKo7gTFAn1zWPx94K4rxuHjVrp21Vbz2Gvz9t/W/6N3bShzOuaiLZqKoDSwNeZ0WLNuPiBwO1AcmhCyuICKpIjJZRM7I6SAiMjBYL3XVqlWRiNsVR2XKwMUX2zDl//ynlSqaN4dBgyx5OOeiprg0Zp8HjFPVjJBlhweTaVwAPCoiR2S3oaqOUtUUVU1JTk4uilhdLFWsaNOrzp9vSeKFF2wYkPvvh61bYx2dcyVSNBPFMqBuyOs6wbLsnEeWaidVXRb8XAh8C7SKfIgubiUnW2e9WbNsPu+777ZBBl96CTIy8t7eORe2aCaKKUAjEakvIuWwZLDf3Usi0gSoDvwUsqy6iJQPntcCOgKzoxiri1dHHmm30v7wA9StC5dfDq1b29SrzrmIiFqiUNV04Brgc+AP4G1VnSUiI0Qk9C6m84Axuu/k3U2BVBH5DfgGeFBVPVG4nHXsCD/9BG+/DevXQ+fOdpfU8uWxjsy5uCdaggZhS0lJ0dTU1FiH4WJt61Z48EH4179szu5hw+D66+25c24fIjI1aA/OUXFpzHYucipVghEjrP2ic2e4+WY45hj4+utYR+ZcXPJE4UquI46Ajz6CDz+0ObpPOslm2Fu6NO9tnXN7eKJwJV+vXla6GDHCkkaTJvB//2fJwzmXJ08UrnSoUMFuof3jD+je3ebsPvpoeO89nyzJuTx4onClS7168O678NlnkJBggw+mpMDHH3vCcC4Hnihc6dS9O8ycCS+/DOvWWfXUccfBV195wnAuC08UrvRKTIT+/WHOHHjuOUhLg5NPhq5d4fvvYx2dc8WGJwrnypWDgQNh3jwb1nzuXOjUyYYG+fnnWEfnXMx5onAuU4UKcO21sGABPPww/PordOgAp58Ov/ziVVKu1PJE4VxWlSrZnN0LF8IDD9g4Uu3bQ4MG1sN7wgTYtSvWUTpXZDxROJeTpCS44w5YtAief95upx01Crp1gwMPhIsugv/+1+fzdiWej/XkXH5s2QJffgnvv2+d99assTaOrl2hTx+bea92tvNzOVcshTPWkycK5woqIwMmTbKk8f77NpkSWDXV2WdD377Wb8O5YswThXNFRdV6fY8fb/NjTJtmy1NSoF8/SxoNG8Y2Ruey4YnCuVhZuNASxrhxdscUQMuWljT69YPGjWMbn3MBH2bcuVhp0MCGN//5Z1i8GB55xOb7vusuG5SweXOb53vFilhH6lyePFE4F22HHw5Dh1p7xtKl8NhjUK2aDVJ42GE29Pl333k/DVdseaJwrijVqQPXXWdDhMybZ8+//BK6dLFSxtNPw6ZNsY7SuX14onAuVho2tCqpZcvgxRehfHkYPNhur73mGpjt08S74sEThXOxVqkSDBgAqakweTKceSa88AIcdZT1zxg3DnbujHWUrhSLaqIQkR4iMldE5ovIbdm8319EVonI9OBxRch7l4rIvOBxaTTjdK5YELE+GK++aiPZPvSQNYSffTYkJ8P558PYsd4T3BW5qN0eKyIJwJ/AyUAaMAU4X1Vnh6zTH0hR1WuybFsDSAVSAAWmAm1UdV1ux/TbY12Jk5FhbRjvvmud+lauhLJlbRiRzJ7ghx4a6yhdHIv17bHtgPmqulBVdwJjgD5hbtsd+FJV1wbJ4UugR5TidK74SkiAHj1sjKnly+HHH21gwnnzYNAga8/o0AEefNDm1XAuCqKZKGoDS0NepwXLsuorIr+LyDgRqZvPbRGRgSKSKiKpq1atikTczhVPCQk2C9/DD1uimDnT+mJkZMDtt0PTptaR76ab7Hbb9PRYR+xKiFg3Zn8I1FPVFlip4dX87kBVR6lqiqqmJCcnRzxA54olEWvsvvNOmDIF/vc/eOIJ67Px+ON2u21yMlxwAbz1lk336lwBRTNRLAPqhryuEyzbQ1XXqOqO4OULQJtwt3XOhahb126p/eILG9F23Dg44wybA/yCCyxpdOlit+POnRvraF2ciWZjdiLWmN0Nu8hPAS5Q1Vkh6xyiqiuC52cCt6pqh6AxeyrQOlh1GtaYvTa3Y3pjtnNZZGRYiePDD+Gjj+D33215o0bQubPdZdWuHTRrZnOIu1In5oMCisipwKNAAvCSqj4gIiOAVFX9QET+CfQG0oG1wCBVnRNsOwC4I9jVA6r6cl7H80ThXB6WLIGPP7bHTz/trZKqXBnatLGk0a6dJZC6da2Ky5VoMU8URc0ThXP5oGpzaPzyiz1+/tnmCc/s3HfQQZY0WrSAmjWhRg17VK++78/y5WN7Hq5QwkkUXtZ0rrQSsSqoRo3gwgtt2c6d8NtvexPHL79YlVVuXygrVbKkUauWDaF+3XVQtWrRnIMrEl6icM7lbvdu6w2+dq091q3b92fm80WL4JtvrJRxyy3WuF65cqyjd3nwEoVzrvDKlLFh0atVs3k2cjN1KtxzD9x2G/z73/bz6qttLg4Xt2Ldj8I5V5K0aWMN5ZMmWdvG0KFwxBHw5JOwY0fe27tiyROFcy7yjj3Wxqj67jtrA7n2WhtW/bnnfCTcOOSJwjkXPZ06wbffWtKoU8eqoRo3hmefBR9yJ254onDORZcInHSSVUd98ondajtoEBx8sHX6e/RR69/hii1PFM65oiECPXtaT/Fp02ycqrVrYcgQqFcPWreG++6zwQ5L0N2YJYHfHuuci6358+G992D8eOstrmoN4GeeaY9jj/Ue4lEU6/konHMubw0bws0321wby5ZZ+0XDhvDYY9CxI5x4IixYEOsoSzVPFM654uOQQ+Cqq+Czz2w2vyeftGqq5s2tX0ZGRqwjLJU8UTjniqdq1WDwYJg92xrDb7zRJm6aOTPWkZU6niicc8Vb7do2X/hbb8HChdbofe+9+e+PsXo1vPACnHeezcuxfn104i2BPFE454o/EbvA//EHnH02DB9uvcCnTMl9u7/+gmeesRLJwQfDlVfChAk2XWzt2vCPf9g+Xa48UTjn4ketWjB6tE3EtG4ddOhgF/2tW/euk5Zm08F26gSHHmrJIC3Nxp369Vf4+29r9zj3XHjpJZu0qXt3G3pk9+7YnVsx5rfHOufi04YNcOutNizIEUfAJZfAp5/C5Mn2fvPm0LevDX3erFn2t9iuWgXPPw9PPQXLl9vdVtdeC/375zxU+s6dVgX25597H4sW7Z01sHNnK73ECZ+4yDlX8n3zjVUpLVhg7Rd9+9qjcePw97FrF7z7rt2S+9NPkJQEl11mJY3Fi/dPCqElj+RkOPxwmDMHNm+2ZY0b2xzlmYnj0EMjecYR5YnCOVc67NhhVVGR+CY/ZYpVXY0dawkEbF6NI4/c/9GokU3aBJCeblVa331n41v98IPN4wG2Xmbi6NatWJU4PFE451xB/fWXlSAaNrT+HfntHZ6eDtOn700c339v1WUJCdCnj4131a1bzHudxzxRiEgP4DEgAXhBVR/M8v5Q4AogHVgFDFDVJcF7GcCMYNX/qWrvvI7nicI5V2xlZNg0s2PHwosvwpo1VioZNAguvXRvyaSIxXQIDxFJAJ4CegLNgPNFpFmW1X4FUlS1BTAO+FfIe9tUtWXwyDNJOOdcsZaQYG0oDz1kd2G9/rqNpDtkiN2qe/nlNkNgfqhawpkzJzoxB6I5FWo7YL6qLgQQkTFAH2B25gqq+k3I+pOBi6IYj3POFQ8VKsBFF9lj+nTr6/HGG3a7btu2dkvvuefaeuvWWYN66GPRor3PN2+2qrHly6MWbtSqnkSkH9BDVa8IXl8MtFfVa3JY/0ngL1W9P3idDkzHqqUeVNXxOWw3EBgIcNhhh7VZ4uPaO+fi0YYNVsp4+mnrBJiUZMs3bdp3vapVoX59G5o981G/vrV7FEA4VU/RLFGETUQuAlKAziGLD1fVZSLSAJggIjNUdb8hJFV1FDAKrI2iSAJ2zrlIO+AAuOYaG9/qu+/gzTehYsV9k0G9ejYGVhGLZqJYBtQNeV0nWLYPETkJuBPorKp7Zl9X1WXBz4Ui8i3QCvCxhp1zJZuI3UrbpUusI9kjmkN4TAEaiUh9ESkHnAd8ELqCiLQCngN6q+rKkOXVRaR88LwW0JGQtg3nnHNFJ2olClVNF5FrgM+x22NfUtVZIjICSFXVD4CHgSrAf8XuJc68DbYp8JyI7MaS2YOq6onCOediwDvcOedcKeZToTrnnCs0TxTOOedy5YnCOedcrjxROOecy5UnCuecc7kqUXc9icgqoKBjeNQCVkcwnFgraecDJe+cStr5QMk7p5J2PrD/OR2uqsm5bVCiEkVhiEhqXreIxZOSdj5Q8s6ppJ0PlLxzKmnnAwU7J696cs45lytPFM4553LliWKvUbEOIMJK2vlAyTunknY+UPLOqaSdDxTgnLyNwjnnXK68ROGccy5Xniicc87lqtQnChHpISJzRWS+iNwW63giQUQWi8gMEZkuInE5nK6IvCQiK0VkZsiyGiLypYjMC35Wj2WM+ZHD+QwXkWXB5zRdRE6NZYz5ISJ1ReQbEZktIrNE5PpgeTx/RjmdU1x+TiJSQUR+EZHfgvO5N1heX0R+Dq55Y4P5gnLfV2luoxCRBOBP4GQgDZts6fx4n/tCRBYDKaoatx2FRKQTsBl4TVWPDpb9C1irqg8GSb26qt4ayzjDlcP5DAc2q+rIWMZWECJyCHCIqk4TkSRgKnAG0J/4/YxyOqdziMPPSWySn8qqullEygI/ANcDQ4F3VXWMiDwL/Kaqz+S2r9JeomgHzFfVhaq6ExgDFGyGchdRqjoRWJtlcR/g1eD5q9g/cVzI4XzilqquUNVpwfNNwB9AbeL7M8rpnOKSms3By7LBQ4ETgXHB8rA+o9KeKGoDS0NepxHHfxghFPhCRKaKyMBYBxNBB6nqiuD5X8BBsQwmQq4Rkd+Dqqm4qaYJJSL1sDntf6aEfEZZzgni9HMSkQQRmQ6sBL4EFgDrVTU9WCWsa15pTxQl1fGq2hroCQwOqj1KFLU603ivN30GOAJoCawAHoltOPknIlWAd4AbVHVj6Hvx+hllc05x+zmpaoaqtgTqYDUoTQqyn9KeKJYBdUNe1wmWxTVVXRb8XAm8h/2BlAR/B/XImfXJK2McT6Go6t/BP/Ju4Hni7HMK6r3fAUar6rvB4rj+jLI7p3j/nABUdT3wDXAsUE1EEoO3wrrmlfZEMQVoFNwFUA44D/ggxjEViohUDhriEJHKwCnAzNy3ihsfAJcGzy8F3o9hLIWWeUENnEkcfU5BQ+mLwB+q+u+Qt+L2M8rpnOL1cxKRZBGpFjyviN208weWMPoFq4X1GZXqu54AglvdHgUSgJdU9YEYh1QoItIAK0UAJAJvxuM5ichbQBdsSOS/gWHAeOBt4DBsOPlzVDUuGohzOJ8uWHWGAouBq0Lq94s1ETke+B6YAewOFt+B1enH62eU0zmdTxx+TiLSAmusTsAKBW+r6ojgGjEGqAH8Clykqjty3VdpTxTOOedyV9qrnpxzzuXBE4VzzrlceaJwzjmXK08UzjnncuWJwjnnXK48UThXDIhIFxH5KNZxOJcdTxTOOedy5YnCuXwQkYuCMf6ni8hzwaBrm0XkP8GY/1+LSHKwbksRmRwMJvde5mByItJQRL4K5gmYJiJHBLuvIiLjRGSOiIwOego7F3OeKJwLk4g0Bc4FOgYDrWUAFwKVgVRVPQr4Dut1DfAacKuqtsB6+2YuHw08parHAMdhA82BjVZ6A9AMaAB0jPpJOReGxLxXcc4FugFtgCnBl/2K2KB3u4GxwTpvAO+KyAFANVX9Llj+KvDfYByu2qr6HoCqbgcI9veLqqYFr6cD9bDJZpyLKU8UzoVPgFdV9fZ9ForcnWW9go6LEzreTgb+/+mKCa96ci58XwP9RORA2DM/9OHY/1HmaJwXAD+o6gZgnYicECy/GPgumDktTUTOCPZRXkQqFelZOJdP/o3FuTCp6mwRuQubPbAMsAsYDGwB2gXvrcTaMcCGcH42SAQLgcuC5RcDz4nIiGAfZxfhaTiXbz56rHOFJCKbVbVKrONwLlq86sk551yuvEThnHMuV16icM45lytPFM4553LlicI551yuPFE455zLlScK55xzufp/T/0SBfxvSGIAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "v8pHXMS8PFHl",
        "outputId": "4b2064ac-1e9b-4bf2-e60a-4879270a1060"
      },
      "source": [
        "# Generating the test data\n",
        "model.evaluate(test_data, test_labels)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "8/8 [==============================] - 0s 7ms/step - loss: 0.8383 - acc: 0.7984\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[0.838300883769989, 0.798353910446167]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_MCmGI2rPSO5",
        "outputId": "630dbcdb-44ea-427a-a5f4-3efc47662305"
      },
      "source": [
        "# Creating Predictions\n",
        "preds = np.round(model.predict(test_data),0)\n",
        "print('rounded test_labels', preds)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "rounded test_labels [[1. 0. 0. ... 0. 0. 0.]\n",
            " [1. 0. 0. ... 0. 0. 0.]\n",
            " [0. 0. 0. ... 0. 0. 0.]\n",
            " ...\n",
            " [0. 0. 0. ... 1. 0. 0.]\n",
            " [0. 0. 0. ... 0. 0. 1.]\n",
            " [0. 0. 0. ... 0. 0. 1.]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DTjsl2MPBflM"
      },
      "source": [
        "**Classification Report**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XVBjWIjqP7S7",
        "outputId": "21cd86ae-91e2-4b8a-e770-c52976e29a82"
      },
      "source": [
        "animals = ['Eland', 'Eland_Front', 'Eland_Rear', 'Kudu_Bull', 'Kudu_Bull_Front', 'Kudu_Bull_Rear', 'Mountain_Zebra', 'Mountain_Zebra_Front', 'Mountain_Zebra_Rear']\n",
        "classification_metrics = metrics.classification_report(test_labels, preds, target_names=animals )\n",
        "print(classification_metrics)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "                      precision    recall  f1-score   support\n",
            "\n",
            "               Eland       0.74      0.93      0.82        57\n",
            "         Eland_Front       0.73      0.61      0.67        18\n",
            "          Eland_Rear       0.86      0.50      0.63        12\n",
            "           Kudu_Bull       0.92      0.81      0.86        57\n",
            "     Kudu_Bull_Front       0.83      0.71      0.77        14\n",
            "      Kudu_Bull_Rear       0.78      0.70      0.74        10\n",
            "      Mountain_Zebra       0.85      0.88      0.87        51\n",
            "Mountain_Zebra_Front       0.88      0.47      0.61        15\n",
            " Mountain_Zebra_Rear       1.00      0.56      0.71         9\n",
            "\n",
            "           micro avg       0.82      0.78      0.80       243\n",
            "           macro avg       0.84      0.69      0.74       243\n",
            "        weighted avg       0.83      0.78      0.80       243\n",
            "         samples avg       0.78      0.78      0.78       243\n",
            "\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in samples with no predicted labels. Use `zero_division` parameter to control this behavior.\n",
            "  _warn_prf(average, modifier, msg_start, len(result))\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UR9dsZL3KgxU"
      },
      "source": [
        "**Creating the Confusion Matrix**\n",
        "\n",
        "In Iftekher Mamun’s code for creating the confusion matrix was the main reason why it was chosen. The code by Mamun (2019) begins the process of creating the confusion matrix. This means that the code must be converted from a numpy array into a set of categorical variables that can be used in a data frame. This creates a column of variables which show the correct identity of the category. In addition, the predictions must also be converted into a confusion matrix. Like the labels they are in dummy format, so they are also converted from an array to a data frame. By comparing the data frames for the labels and results, Matplotlib can construct the confusion matrix. \n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "AVDcXCUkQ41J"
      },
      "source": [
        "# Converting the labels from a numpy array into a dataframe\n",
        "categorical_test_labels = pd.DataFrame(test_labels).idxmax(axis=1)\n",
        "# Converting the predictions created by the model into a dataframe\n",
        "categorical_preds = pd.DataFrame(preds).idxmax(axis=1)\n",
        "# Building a confusion matrix that will compare the dataframes of the labels to the dataframe of the predictions\n",
        "confusion_matrix = confusion_matrix(categorical_test_labels, categorical_preds)\n",
        "\n",
        "#To get better visual of the confusion matrix:\n",
        "\n",
        "def plot_confusion_matrix(cm, classes,\n",
        "                          normalize=False,\n",
        "                          title='Confusion matrix',\n",
        "                          cmap=plt.cm.Blues):\n",
        "# Allows the results of the confusion matrix to be normalized\n",
        "#'\"prints pretty confusion metric with normalization option \"'\n",
        "  if normalize:\n",
        "    cm = cm.astype('float') / cm.sum(axis=1) [:, np.newaxis]\n",
        "    print(\"Normalized confusion matrix\")\n",
        "  else: \n",
        "    print('Confusion matrix, without normalization')\n",
        "\n",
        "# printing the confusion matrix\n",
        "\n",
        "  plt.imshow(cm, interpolation='nearest', cmap=cmap)\n",
        "  plt.title(title)\n",
        "  plt.colorbar()\n",
        "  tick_marks = np.arange(len(classes))\n",
        "  plt.xticks(tick_marks, classes, rotation=45)\n",
        "  plt.yticks(tick_marks, classes)\n",
        "\n",
        "  fmt = '.2f' if normalize else 'd'\n",
        "  thresh = cm.max() / 2.\n",
        "  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):\n",
        "      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment=\"center\", color=\"white\" if cm[i, j] > thresh else \"black\")\n",
        "  plt.tight_layout()\n",
        "  plt.ylabel('True label')\n",
        "  plt.xlabel('Predicted label') "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mJXoT_cyERYN"
      },
      "source": [
        "**Plotting the Confusion Matrix**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 328
        },
        "id": "j4QSubrrXwiu",
        "outputId": "b634065c-6249-4663-8775-198c3d42a48a"
      },
      "source": [
        "plot_confusion_matrix(confusion_matrix, ['ELND', 'ELDF', 'ELDR', 'KDUB', 'KDUF', 'KDUR', 'MZBA', 'MZAF', 'MZAR'], normalize=True)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Normalized confusion matrix\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tMeavnVxuxB3"
      },
      "source": [
        "**RESULTS**\n",
        "\n",
        "**Control Validation Accuracy and Loss:**\n",
        "\n",
        "![image.png](data:image/png;base64,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)\n",
        "\n",
        "**Control Confusion Matrix**\n",
        "\n",
        "![image.png](data:image/png;base64,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)\n",
        "\n",
        "**Control Classification Report**\n",
        "\n",
        "![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZwAAACmCAYAAADu1zlGAAAgAElEQVR4Ae2dPZYUR9aGZyuwDmkPwA5gA7AA5CN8XCEX5A62THDGmrFgzhkLrEEeyKO+85S+t3n7TkRWVndVVv68cU53ZPzHfeLGvZlZlVl/+/PPP3f5C4PoQHQgOhAdOLcO/O3cA6T/KHF0IDoQHYgOoAN/2yWEQAiEQAiEwAQE4nAmgJwhQiAEQiAEdrnCiRKEQAiEQAhMQyBXONNwzighEAIhsHkCcTibV4EACIEQCIFpCMThTMM5o4RACITA5gnE4WxeBQIgBEIgBKYhEIczDeeMEgIhEAKbJxCHs3kVCIAQCIEQmIZAHM40nDNKCIRACGyeQBzO5lXgdAA+fPiw++GHH3avX78+qtObtjtqkJVUfvfu3e7OnTv7v59//vmaVKRVRr1Tha9fv+4eP3581fex63uqeaSf5ROIw1n+Gs5Ggps6jpu2m43gF5gIRr86HKYh53BKh9Mb648//tjdv39/74hOOd4FcG5uSNbryZMnuy9fvkwqexzOpLgzWAichkDPCZzD4eDY6lUNJwn37t3bYbhwOnE4p1nXqXqJw5mK9EbGwUi8efPm6gwUo8AZKQFlo5y/1i0Yz+dWCkZMgbZqQyxD422Upza6glE76ioMteuNRX9Pnz7d/6nPOqb6P3esubx8+fKKixtnl28sy8qrtkOmKRyOz12cXY+Yh65yjuFP3bt37zZ5IfuPP/54Vfbq1aurJfT5MI/Pnz/vy3wN1K+3e/78+VV/sDznWX2d/7Nnz3bfvn3bc3r06NHu/fv3+znDTWmOudpgv2r+akd/P/300w79UpnLVlm+fft23z97lnb0KZ7I/vHjx92DBw+ueGhdNd4V7DMd5ArnTGAv3S2bk89TUFgCaRlClBRFk5EgllGjjurRztPU8z6rjL2za8bWWLUN6VY76rtxQw7OqIn5Yx70S/D5t/o/Z57mImakHz58uDcwzo45ePoQS59zix99SX6v22Lp5Tc5ZhzJV9sf63CYHycLcKqBPIyjjKaXMweXl/nIeaid5kgaY45DIk/59FfTPsYpjnFuLV13B8M4nuYYJyB5vEyyITuOy2VjHNq542XP4NTgjBNTn0qLLW1zS+0UK54+9gSqkUDBtGE5loNxXDJWOutRrHa1T2/LsdrTvwc2OX2pHy/rtauGwftm08mo076ma//nTPfG1nzFULEYDLHE4GA41IbYjSbykFZfLp/GrWvgdY49HjPXOp7WXDLI0DE2/R0rU3VS4o6x9WOXDRYYVc1BMeNjvM8RkJsrkXrF4E6EcT3tx5qTHBeyyXlShkxcueBUGIs/BckLaz9WucesVxyOE8nxrQhUI4GCkUfguOdw6sb2SdQ+vYxjlJx+q/FRPfLZ9HXsVrvWZlLfMjBsVEJNa7wp4t7YyHRTlnDWWiFDizt8vI5kbbFU2U3j1vjqizXAOfbWXPVaMTKgD5JjSKbKUtwPORyMM3WnDvDA8aCz3MKrTsXTfqx5xuGIROJFEHAjUY0Qm6EafQlFu14Z7fw2l9ooruMo32M2F/0TK7Ta1bE8LWOjPmpa/U4RD419E5ZigfEl0H/rq+ZDxhm+8DpVcF2qfbIGN3U4kg9ngtzISl/6nMPHYg78KXhaa6BbS6pDjOGGxzk/t/Hx/Bg2XEUwL465/aUrPebFuiIrZfo8h/bIo6saP6aMddWVCcd+S83T8KSexvN51TFq2TnT+QznnHQv2DcbUrcQiGXAmBKK2XMqMnjelvoK9FPLWm2oQ7tWmfprlakd4/lY/tmRDAwblVDTmusU8dDYLfkkO3Nz+Vxu6ogxBvjFixdX61fbUI+1bo3lfd6GBf0zrgfYMzfNk1gG1OvVYxletattkF0fjlNHH5AjHwZU7dyJaA1aDqe2o33PCNe5Hps+NBYMJdsvv/yyvwKWw8FxSDZngmz60J9ymLuc3qe301yGZMXpaT71FuCxso+tH4czltTC6rWMxMJEyHRDYBME6hWOC43D0dWO5y/1OA5nqSt3YN5xOAcApTgEZkIgDmcmC5Fp3JxAHM7N2aVlCExJIA5nStoZKwRCIARCYBMEckttE8scIUMgBELg8gTicC6/BplBCIRACGyCQBzOJpY5QoZACITA5QnE4Vx+DTKDEAiBENgEgTicTSxzhAyBEAiByxOIw7n8GmQGIRACIbAJAnE4m1jmCBkCIRAClycQh3P5NcgMQiAEQmATBOJwNrHMETIEQiAELk8gDufya5AZhEAIhMAmCMThbGKZI2QIhEAIXJ5AHM7l1yAzCIEQCIFNEIjD2cQyR8gQCIEQuDyBOJzLr0FmEAIhEAKbIBCHs4lljpAhEAIhcHkCcTiXX4PMIARCIAQ2QSAOZxPLHCFDIARC4PIE4nCOXIOvX7/uHj9+vLtz587uhx9+2H348OHIHg5Xf/369Y6fiE4IgRAIgTURiMPprCYGH6eiv/v37+/47XEFjh8+fBiHIyADMazgJ5bv3r0bqP29CMerNpU/ffTKvvcw7yPXsbEnGJUljBT8ZMjzVb6EmHnfvXt3v7ac2H358uXgtJH7yZMnV/oAy2/fvl21U59j+7tquPCD33777Yole+XVq1dXXCqzt2/fTiJtHE4HM0o7tGnjcDrgSraMoJzMWG7UdydDGoNBf/TBMTGBtRprsMv0LpZEt3zOh/SNiYql9FJp2MACXr///vuejepcTMAbDKw1lpOpjHpd1vV//vz51d7lWGy35HC48/LTTz9dOWz0A6f8+fPnva48ePDgmgPqsT11fhxOh6iUtFO8X7TWFQ7tdOZNzCYioABPnz7d/6lcZZSzuZRPTD9rCDIiGEcCaeQ7ZBAp9zrVyTibOoaXzfEYWVx35CzkUHtzbjGonOSEnF2vvznla97aE6QxkDhRjGQvUI995be26YP9U69ytuRw0BV3KlztSP4Wnx7fU+fH4XSIVsfhZ9s0qUaj1Y2cDJuCYz7zkSNxI4lxcGNDWvVa/S4pz2WRnC9fvjwon/NBXtr2PjODFeVLCegCDgcd4vjevXu7N2/eXOX15KhGmfbopeuK6iyJB/L6fuIYYykmMBoKvv7Ij6OScVU76Z6unpS/5lgsOMF79uzZlQPmqu/Fixd73aGMffX+/ftJUMThdDC7Ereq+AbxchSbRdSfHJUbGeor/enTp/3mwMAq0IcbEeUvMZYsyCOZlHdIHuqLIxsEIwJ3D/TlztrL5nqstceg9vSjN3faYiDgQlu4wEBh6Q4HJjhgDCBr/ejRo2tXL5LTYzkomMBGJzS5wvnrtpkcj5wwDgfd0ZUjtgcnPYUzjsNxzbVjjJ1vZCvaH7YcDsaAzUJMkGGhrh972dodDsqMIXCWh9ju4ZV/8OPWCZtHYYnOhrmjD2x4d5T1ik4yHoph6ScrS3U4mrcbQu0ZGcZDLFSOXri+kS9dmcKoah6XjKsTQedw3jhyfcYlhwxnyo7lfBP54nA61A4ZRRbQ78PTDYvMhqGMQB9Ka/OozNPU4099YKCV3mcu+F/lhNzulF3maiQkNn3A0Q0rfNxgq+5SYuYveWVslUYGyTwko4yoO+FWX0thgjyu9xx7GtkOfa6DjnA7rhpPsdqSw3EO7Dv2EA6nOiM+34GzHNA59SUOp0OXBcDw60+OQ4ZA+cT6bEGbXWV+G8gdDEN62vvEwPBNI99onSkuJhtZdRtIrHzybACYtQyu81Ub70+sid0hqe5c46orLjtzlk5Uh0M9yew6UvtTnSUxQW7fdy4fZT2Hg4y9r1LDS2ViQr9TGNdL6179WrR/9ZmrHHFBx6ZyxHE4l9aKjB8CIRACGyEQh7ORhY6YIRACIXBpAnE4l16BjB8CIRACGyEQh7ORhY6YIRACIXBpAnE4l16BjB8CIRACGyEQh7ORhY6YIRACIXBpAptyOHzVtD47M9UC+NdW61eD/Wu+9WuwU80v44RACITAuQksxuHw3Xk9qyDjXb+nfwjWKRyOOwd9r/+YZx2G5kA/a3Q4eqbkWF7+zImeg9Ia+zpUB646c4/RXzEZq8uVZd0T6o94ibrkz82MfT5Ez+dIdljqORt/RodyL5u7ftx2fvU5HP95Ai8by/m286H9Ih0OSjl2gzqkIWPv9YaOMXR68SL1jnUSQ3M4tq+hec6lTCcHyEYYkt/nTH13Ms6m9lHXxPuZ63HVYfRZzqM3Z7FUPaVh48e99nPP1xrrIcTKqDd/2Lk90Ktban30Rq93qWVrS7Mnej9PAGd/d1reNNBYfW1IKSUbjFCNTyuN4dLZj86GW/XG3G6rxo35SNlrWU235uuiVtm8bKnHVSbSrIWMZk8uyr0O68WZGHE1RKTpk76XEFq6h44euiJxBpJTnJbucOr8SR96jQ0MqDfm5wmoy36c6p1hWp9LxehK6+cJ/vvf/+656q0DYzmfSo5FXeH4q2IEoLV55TgoYyPLEHldP6avmlb/NUZpcVpyYH4WXh1MTR8apxrnOvYS0+4cOMao6m2+Q/JUFrTVyYJOPmjPsf6os4TgesHxKX6eQAZberkkB8ya+f7j+JQ/T8AVj7jI0C5BT247RzkTZNfPE8BWV3ni/Pe///0q77ZjHmq/KIcDODfwCOeKWtPVaHldP67thqC5saAeY2hOtaymD41T5zs0j6WUyeHIKTBv5R2SgTYyFH6yQT5OC+clJ0Oejg/1e+ly6cU5fp5AsqFLctDKm3Os/XiunydAdsbAkW3B6UhWPreR42G/fPz4ce9ccDLYLf8ZCI7PHRblcDAo/PmtBykqm1hKpSucasC9rh/XdkPQZSxoX9vVspqu9es4db61fIlpZMJpuDO4iXOApX6egL7ok74JOrtXeu6c0B02u+vxTdceli25q37PnYnWEC5607P2j9JjZZCdaNUfKmvVX2oeOuGf06APXNn885///J9blXCe6lbj4hwOCsAm44+gzatNR77O7ACpV+FLoVU21G5IybQJaE9gXL/CqeOpTH3STg5ReYrpy42Q8pccV3l9TSQXclenpDLiulZDa+Dt5nyMnsoJSzeVdpmH9IH6vfKhsrlyYc7a18yRY0/D6dDnOuiSv5bfZUWPtnKFUzmwZ7BFXMWIs77Jxy1HOCvtzE59vEiHow2qzQZA3XrhVot/iOhlvPa/V1bb9UCzcDgtjVcdCgunMm4PMEcUXUZTZcRyfjK4XibZevNYUr4zk8w+f8nPWik4r6E2MKtroD7mHEuHteYuO/OW/FUPXJ+rMaau+qvt5szC5+b7x+WjTs/hoD+9V+375zfo0RS3jVyeSx77V5/RC7+VCBcx0+c7U8x1MQ5nChgZIwRCIARC4HwE4nDOxzY9h0AIhEAIGIE4HIPBoW5l6NaEx61bO6V5kiEQAiEQAh0CcTgdMMkOgRAIgRA4LYE4nNPyTG8hEAIhEAIdAnE4HTDJDoEQCIEQOC2B1TgcffW2fr30tLiGe2MOPGPD50AJIRACIRAC1wl0HY6eE/DvwnN87mcecBg+5vXp9lOndDjqy78wwDHf9x8KcThtOvWLGIc4qhd/5sT1zvO1Rkv8Qgd6rvmP1fnK0k+wvL8x+irOc4qRR8+H8CyR3hw9NEc9n+Ms9RCj90f5lp7Fqc/hzPrnCVhEFJg/lJw/jlECjs8VUBDGmVPAQI6ZUxzO/66aTlzkZNCd3psWvDX13cmQ7j3MuETuVc/RL/KGgliqntJi622HeHm9OR1rznIylVFvrrDz/clDjWKkPuSAen2sLZ89saifJ8AwsMF5Wh5F4M+fnGeBWGSdVbhx0CJrET1NG57q50yDtjIiAFKe+iSW4tRytavzYJ4KkoEXP6pP9ac6h2L1QaxQ56IxyedNBshXx8M4UAZDySkZ6Bt+auMsNeZSY9hITmQg7evak4t18rVqrYPaolNeV/lzjZHFna7W3zm15t5iUDmpHfluhJU/17g6T9KHXmODLNpX7D0FdAzZcTLisDWHg67wGh9d1XC1g37N9ucJpNz//ve/d7/++uv+jzeN6tUwLKgrNAurDaNFlgJ4mjYYXBSkKhn1va7at2L6kaGnvNVX3ciMeexnLHUccSEmeJr+kU1cfDzNT4yUdhkkJwz4W0Pw9eQY+XHIYtSTES5iRR3aSm+8jTP2/Dkf+5w5PsXPEyCv9B2HvrSTFuYuJ8wxxpKTM+UNrSe6pP3CvsJRoTtcKZGvW3Rwoe5WnI9YILdeXwPbWf48ARNj0YhZNP4QAIfzr3/968rxSBF8E1HXDYqnXTloW9NeV30TMw+/CqhnyS0DThtX2Jr2/lvHOlPyMvIY2/9kCJ0BbcSL/Nb81K/KvE/np3pLjLWeyCOZlHdIHuqLif88gbejDv0tKUhPMKhyDMpDR4cC9dA3uNAWLi350VP1PdTfXMq0N2GCA/bX5iPzUKAtDgomsNEJTXUsqtfiNdT/Essk62J+noBFxrlgDBVkQC/hcNxgMZ9qaGS02WgKUmIpbE2rXiumP8agjYeWE1I54/gVlHiR35qf2rEB6tk8Y68hyEH7Jq9rN0bOlj4uzahKTnQKZ+BrjiyeVt1DMSxpW4PrXi2bY1r7Ay76OQLtJ6XHzhtdc33zdv75juev7RidWNTPE7Q2uCsxiu5G0dNuQOmHsw7VJXZlqOnWxpMyqp36VBplUR3ffNXB1PSQktG396+6GtvH8TJ3OC5La35q5+xkjMRLdZYaV+bw0084SCY4cXba4k0dMXHm4tlro77nGrvet2SRzENOCNl75bBa0hUO64Q8rve+LyiH06HPdZCbq52Wk0L3YLKFN0ZXDi67OOsKECcMa6XPuWe6X4tmgq0rHBQcYbRJdMvDFV+bRZf9XCZLkYjdSNR07Vd1GVNjoTS6lVDrqw71q7Gr6R5YZNdtC/VH3JoL+ZK9tvMNr3kyrxq8nd8SqPWWmq7ykfagtRVfylyHYNJq43y9vyUcSx+kXy67yy/dkkzUUxvtKa+vstpO7eceI5NkcPmYN8xaDgf90ec0yK1vudEGXipDj7bgbLTG9WvR+XkCkUkcAiEQAiGwegLdK5zVSx4BQyAEQiAEJiWwSYfjt2t0+a64dftm0hXJYCEQAiGwUgKbdDgrXcuIFQIhEAKzJhCHM+vlyeRCIARCYD0E4nDWs5aRJARCIARmTSAOZ9bLk8mFQAiEwHoIxOGsZy0jSQiEQAjMmkAczqyXJ5MLgRAIgfUQiMNZz1pGkhAIgRCYNYE4nFkvTyYXAiEQAushEIeznrWMJCEQAiEwawJxOLNenkwuBEIgBNZDIA5nPWsZSUIgBEJg1gTicGa9PJlcCIRACKyHQBzOetYykoRACITArAnE4cx6eTK5EAiBEFgPgTic9axlJAmBEAiBWROIw5n18mRyIRACIbAeAnE461nLSBICIRACsyYQhzPr5cnkQiAEQmA9BFbjcD58+LDj56Ffv369ntWJJCEQAiGwIgJxOCtazLmK8scff+zu37+/u3Pnzv7v3bt3o6bKyYPa0J5+CF+/ft09fvz4qowTDU44lhZ+/vnnKxk4HhMqSz/B8v7gNpbzmHGnqoM8d+/e3XNhjb98+XJwaPThyZMn11h++/Zt3w4G6g8mMFLZwY4XUMHlY498/vz5f2b9/Pnz/cn4+/fvr8oqs7dv316VnfNgNQ7nnJDS980JyDnI+GEwHz58eNBBUN+dDGkMEP3VgJEaa7Br20ul65yZvzuP1rzEUvWUFltvM8TL683pWHOWk6mMenOFna8/BlaMvA269+jRo50bXi9f2jHy4GjlZJAbDu5Qf/vtt/2+Yc9Jbto9ePBg9+rVq2t1p5B/EQ6Hs9enT5/u/zibffPmzd5jywABWWfCrc3n5W7EyKcv8mjvZbpFp35bCtxaIB+LtpoP+fwpVGNR2y31rF3yKZYRkaMgDZdDPCn3OmwS1pu4Bth53Vo+tzQyuNMlje5Jn3vzbTGonNSWfNc35c81rvuBNMYULjKorblTD9vgV7joGLK74aUtdXA4Q/21xlhKHnLDTA5b8iqWw+nxmULOxTgcfT6DIqGEQGPTygBVhRU86vc2svpSH6TZqCzQvXv3rpSYco2pfsfE9MNmYG4c+3w9zZiaY0+OMePNsY4bPsn58uXLg8aQ9RUT5KKtO2H4kcZ5eb05Mqhz8rXnGF3jxMf1o7YhXXVDeoneEpSGCfpKeimBucoJc8wZuJjAaCho31IHRhhddEKGlzN/mPA31a2jofmeq0xXdjhacUBeePqVHfVevHix1xGYsI/kjM41N/W7GIejzSjl8k2LMHUzkudKLIE9Vl+ex7HOADy/V9frcIxhlHIT+8anD/pWPerqWEZTcqhsX2HB/5ADufUneUkfCtQRSzYIjFjTGhhD/GrZHNPSXV1dI5PyWvK5DNSTo0W34NLSFfTMdc/7mOOx9ipMcMAYQPIwlMg8FKiHg5Lx1AlNvcJRvTU6He0BOVlupZEHA3GUU8HhoBu60kNX/MpoiPVty+JwGt9qu6nDYWP4lVE1IuoXp+K3AdyIsGnGGOPbLvxU7ZEZmdwoIp+nx8wFRrparPXZUD1nVOvOIc182fDuJOHk6bHzhCVta6g6VsvnltaJlhtC7R8ZxrFzRrd6+jVUNrb/udVDJnRHzkaOVSdrHj979mxvX2gjhwznqW41rtrhoBhsyJ4B7xk+FqA6Dk/3FI6N72eV9O9pNhV51OPMVEF5Sq8pRvl1qwS5Klvy4FGdkjOQgW4ZVuppw8F3KcF1T8YWORQk85ATGpK76qL6nXOMPHBR4NjTcDr0uQ5yc7XTclIyxGu6wuFqxZ2N2HmM3H5LDUZ+RcPVEJzlgLztqY8X73A+ffq0B+5enGMZJ21mlbsDALJvcodLe7Xx/rxOPa5jtW4DMZ5/FkEfGGHdJtGYvtHqOEtLu3xVdmQRa18LGVx41DbeH+VLZFV1xWWHieSvDod6LR1RfZXVdkvRGdZSMtR17Tkc9Edffa7GF4Os/tAj3VZaCo+hebIPfvzxxyv5JGd1qNXh0CdcesyGxrxt2SIczm2FnHt7NpYbHBRpzBXV3OXK/EIgBELACcThOI0LHOtM1x0OZ2x+JXaBaWXIEAiBEDg5gTickyM9vsN6i6jeQjq+x7QIgRAIgfkRiMOZ35pkRiEQAiGwSgJxOKtc1ggVAiEQAvMjEIczvzXJjEIgBEJglQTicFa5rBEqBEIgBOZHIA5nfmuyuhnVZ0T4Ft6Y4M+c+Lf29M0+PXew1C9ZDD1z0uNTWfq3G70/2Izl3BvrEvnIc+zzIXo+R/oABz3E6M/oUO5ll5Dv1GO6fOyR1gOvPHNTn0GqzOqzO6eep/qLwxGJxGchIOcg44fB9DcP9AalvjsZ0r2HGTFSGJIlhTpn5u/OoyWLWKqe0mLrbYZ4eb05HWvOekVLZdSbK+x8/TGwYuRtWg9AevnSjpGHNwbIySA3HORskSc/T3CDVeVrw7xHiz88NS/4I5YBQrl0dkNclY1FULkbMfL1AkXKvax+Vbn22RPDx6JPGQPy+VOoxqK2W+pZu+RTLCOCvATSrTVSfcXwduZsLtabuAbYed1aPrc0MrjTJY3uSZ97820xqJzUlnzXN+XPNa77gfSh19ggC/X8vYTkoWPI7oaXfPb0VO8MuwRn5PZX1khexXrLQo/PFHNexBWOjL82EZsTaHqDtIOqmxLF621kytzJkGYMxvMn/WUQGPOYQD964STHPl9PM6bmWDfeMePNsa7WjLlJTr3Nd2i+sBYTtXUnDD/SOC+vN9TnXMp87TlG1zjxcf1ozbXqhvQSvSUoXU+eWn3NLY+5ywlznJ8nOH6FdGWHo5XD5lYZPP1datTjtVvYPnSFfSRndPyox7VYjMPRZnSnoDyME+D0J8PkStzCor5qGf1pE6usV1flijGqmgdxdWhyWtTjjyBDjJLIqKhM/S41Rg7Y6U/yVr4t+agjlq330qmN81PenGM5HF1do6fK43goUE+OFt2CS0tX0DPXvaE+51CmvQoTHDAGkDwMJTIPBerl5wmuvzE6P08wpDEHynwzYoTYYMr7z3/+s99YMuRSXMr9uDWE+qplN3U4jOlXRpoj8yCoX5yK3wagnowIBnaMMa5znmsamZHJjWKP+5AMMNLVYq0HX65yxLmWzy3NPHEGfmUGJ0+PnTMsaVtD1bFaPre0TrTgos8ktH+UHjtndM31zdsNlXm9JR0jE7qjz77QLzlgnbApzs8TjFhZKR4gZayU949//OOakQe+rnDomvr8tYL6qmX0XR2Hp2t9pdn4flZJ/55mU5FHPc5MFZSn9Jpi1ky3SpCrsiUPHtUpOQMZ6JZhpZ42HHyXElz3ZGyRQ0EyDzmhIbmrLqrfOcfIAxcFjj0Np0Of6yB3fp5ABP+K0SW/pQYj/6wnP09wndfV1QzgtFHlcJQnD85ZsF89aDOr3B2A+irD7ZMsitoQkz4U6lit20BsKneI9IksfoXDeL7RDo0793KXr8rO3MW6ZXBhUdt4f0tlVXXFZYdJz+FQD5mr3KqvsiFHNWd9Qe8lQ90DPYeD/vS+Ss3nFeoPPZrqs4opGLMP8vMEU5Be2RjV8aFIY66oVoYh4oRACKycwCK+NLDmNdCZrp/hcsbmV2Jrlj+yhUAIbIdAHM4M1rreIqq3kGYwxUwhBEIgBG5NIA7n1gjTQQiEQAiEwBgCcThjKKVOCIRACITArQnE4dwaYToIgRAIgRAYQ2B2Dkcfoo/9WidfB/XnPMYIfWydKcY4dk6pHwIhEAJLI7B6hyMHNuY5mt7ixeH0yIzLr8+IjF0Lf+bEv7WnNfXnK/jixdLC0DMnPVkqS/92o/cHm7Gce2NdIh95es/U9OaDPvAgo/QBDnpxJwzUH+Ve1utvSfkuH3uk9VYGnkWqzyBVZvl5gpGrfsgZyDjdZvMdGmPkVDdZrfIfy5L1cidDunfVi5HCkCwp1Dkzf3ceLVnEUvWUbun2EK9W33PI05z1ipbKqDdH2NXnUtwAAAjvSURBVPn66yWWtT6650/c1/KlpZEHRysng9xwkLNFntX/PAFnmq13XgFCG4NjnY24EaFc+a5AgNPmUjkxBol3qXFLjTcQq0wbklh5iv0rxywYfahM7Rivlnm7IcV02ejXZXaZJI+Xax7EY8cbmsscypCvtcbOujVPyr0O60E/xDXA1evW8rmlkcFvA0vXnFNrzi0GlZPake/6pvy5xnU/kD70GhtkoZ6/WYQ8dA7Z3fCSj23C4chAz5XFTeeF3P7KGsmrWG9Z6PG56bjHtDv5LTXfFNoMrhTK0yRrmvzWZkGBtIF8w9bNClw2M/mEqsga1+ekelJc9cnCEHw8tR8TMxc53zovTyOvjE1vvmPGm2MdX0vJmZ8n+K6j6AFvlcjPE3x3wuy3/DzB8btZV3ab+nkCN+RczvE+sY8fP+4N6qdPn/axn8lzLEcixG6klFcdDlcmOITqDGq6Z8DZ6FxF1LnQJ39yAIxf+9ScWjFz9z79thAyyIlRjz8CscbTfFXWGmNJecihtdM6K++QHNQXy9Z76dTe+SlvzrFONvLzBN9XSXssP0/wnckxR9oDuh25qZ8nwDhgWH/99df9mRtKxJk+l7K6ihiC2TJI9CfjQ0wdghSVTdxKy4DL0O8r/f/lta4+lKeYunIArT5Vr8Y6W9VcZFiYI4F+MaLulMmnnjs/Geba/xLTWjetFzIgn6fHyAWj3nrBl/US5zH9XbIO8+RExHWs6tzY+cGStjVUHavlc0trn8JFt7y0f5QeO2d0q6dfQ2Vj+59bPWRCl+Rs0K9N/TwBALQR2AwYCpwQgXzfaK3FU3uVDW0e4Nb74Z7WmPTpQZu+5lMHRdfLM7URxnymgqz1isbT9CUu4qH5tYyGz3epx3V9nK1kQnY/iVC+Yq1VjxFreEin1NdcYvRAuicdU5o5SuYhuYbkrro4F7mH5oE8cFHg2NNwOvS5DnLn5wlE8K8YXfIvS8DIP+tZ/M8ToDgy0No42kzaXH61IkNCHc/nWApHnVpGffp3B1PTIMfI6QpC86r59O3Owefy+++/j7oyq7K1bgPRr8+hNQ+X+y+VWfb/Hn9JpbWFjYL0BhaHeElH1HYJcdUVl535S/7qcFwvXW7V1x6p7ZbAhDkik2Rw+SjrORz0R199Rm6d6dOGzzXUH3qkD86XwmNonuyr/DzBEKEblmlzyjHRDcfuIG7Y9SyasbHc4LSuAmYx0UwiBEIgBG5B4OTfUrvFXLpNdfbmDgcDvdSzOBdUztQdzpqcqcua4xAIgW0TWITDYYkwwro0Jl7L1Q2y+S0nZKu3kLatopE+BEJgLQQW43DWAjxyhEAIhMBWCcThbHXlI3cIhEAITEwgDmdi4BkuBEIgBLZKYPUOx79m6V862OqCR+4QCIEQuBSB1TscwOqbYHE4l1EzfctQX/oYuw7+zIl/SUTrqf6W+iULPxmqz5z0Vqqy9G83en+wGcu5N9Yl8pGn90xNbz7oAw8ySh/goBd3wkD9Ue5lvf6WlO/ysUdab2XgWaT6DFJllp8nOOGqy0AtcQOeEMNFuqrsMZj+oG5vUqyVOxnSva/BY6TGGuzeeFPn1zkzf3cerfmIpeop3dLrIV6tvueQpznrwc3KqDdH2Pn66yWWtT6650/c1/KlpZEHRysng9xwkLNFntX/PMGhRdMm0dmIG5X69WAZGPJ5PQ5/eGrezUZMuV4ISp76dOVjPhqzbsyhs8U6l9pnT07qaR7EGpN876POqbZb6ll75SIjgrwE0nCR0az1labc67BWrDdxDbDzurV8bmlkcKcrPZS+9+bbYlA5qS35rm/Kn2tc9wPpQ6+xQRbqYRfYrwroGLK74aWMOvl5gu/vdKx8xO+c8eS31KQMY4RCaagv469NhJMin03L7+GQ1matm5lxqjIrzxW1Kq7GHjPPXh3mzRj0zTHzZX4ETyOX5t+aa6//JeRrzZir5MzPE3zXBfQgP09w/SW87JH8PMHxu1tXdjgS7AgOm1tl8PQrO+rx2i3sJid/9Xbb8SOPbzG5w2GDIaBf2Wi6gBEEXSVgpNw44wg8Tz/ARh2F6ixaRlzz0DiKcWQExiCPvo4Jaqf+XE6fF/X401jV4ajsmLHnWBc5kFt/kncMV+qIY+u9dJKXMcRPeXOOpc9clUs/lKcTkt78XW9pC5eWrqDH6rvX15zykZsTMpjggHnnmQwlMg8F6untyNgWndDUM3jVm+rziqE5n7pMe0C3Izf18wRjYLL4bAi/deRGiT5IA9I3Y827jcPR1cfQfNm4GL0xBo156i3T9OnzJk1fzB8H6FdX1IODjCt11hLEz42i1vAYGWHUWy90ifUhXkKQ7rtOwcnTY+WAJW1rqDpWy+eW1kkhNkGfSWj/KD12zuia65u3Gyrzeks6RiZ0R84G/ZIDlk1R/OzZsyu7KocM56luNU5+hVMXUhtGCidFkREmLcUDpIyV8qrDUb4bH/XtG1ObXuPVeXmaumMMGv37WSVz9TTzkLycmSooT+k1xbDzzytYH3fKyAo3NkRvLbRWvn7OiHY3Mdbex9TH0mPGlX66/JJ5SK4huasuTi3fTcZDHrgocOxpOB36XAe58/MEIvhXjC75LTUYwVEOavE/T3Bd3OsplEreltgVChAqw1DrdoE7EerTh/L0GY7a+RWTNrLKFDMOgT78ykLOodVOba5Lcz1V27VuAzF3n2NrHszTuVwfZXkp51xlRxqtO2wUZHBhUdt4f0tlVXXFZYeB5K8Oh3rSY9cR1VdZbSeuc4+RSTK4fMwbZi2Hg/7oq8/ILUNKGz6vUH/oUX6e4C8NgEuP2Tl15OJXOLcVjo3mZ9C37e8S7dlYbnAwqPUq4BLzypghEAIhcEoCcTinpHmDvnSm6w6HMzZdbd2gyzQJgRAIgVkSiMOZwbLUW0T1FtIMppgphEAIhMCtCSze4dyaQDoIgRAIgRCYhEAcziSYM0gIhEAIhEAcTnQgBEIgBEJgEgJxOJNgziAhEAIhEAJxONGBEAiBEAiBSQjE4UyCOYOEQAiEQAjE4UQHQiAEQiAEJiHwf2dYdgTLLyJ8AAAAAElFTkSuQmCC)\n",
        "\n",
        "\n",
        "**Control Summary:** \n",
        "\n",
        "The control model ran for 30 epochs and achieved a validation accuracy of 91.88% and a loss of 0.271. Overall, the training and validation accuracy of the model began to plateau around 19 epochs. The testing accuracy for the 93.1% and a loss of 0.3344. The model produced a precision of 0.92 for the eland, 0.90 for the kudu bull and 0.97 for the mountain zebra. Overall, 92% of the images for the eland were classified correctly. 7% of the images were classified as a kudu bull and 1% were classified as a zebra. 95% of the images for the kudu bull were classified correctly, with 4% classified as the eland and 1% classified as a mountain zebra. 93% of the images for the mountain zebra were classified correctly, while 5% were classified as elands and 3% were classified as a bull kudu. \n",
        "\n",
        "**Experimental Training and Validation Accuracy**\n",
        "\n",
        "![image.png](data:image/png;base64,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)\n",
        "\n",
        "**Experimental Confusion Matrix and Classification Report**\n",
        "\n",
        "![image.png](data:image/png;base64,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)\n",
        "\n",
        "![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAdkAAAD2CAYAAACayv1VAAAgAElEQVR4Ae2dPZI0RZKG9ypwDrgDcAM4AHAA0AF5kRllBUAdpLFdRFB3bSTAbCRGAw1Eeu1peJq3ncisrOrqrowqD7P64t/D/Q2PeDPy68z8j7sOjUAj0Ag0Ao1AI/AsCPzHs0htoY1AI9AINAKNQCNw1yTbTtAINAKNQCPQCDwTAk2yzwRsi20EGoFGoBFoBJpk2wcagUagEWgEGoFnQqBJ9pmAbbH7QODbb7+9e+WVV+6Ijwkff/zx3Ztvvnn3888/H9Pt5tr++uuvd++99949xq+99trdd99994DBWt1DoxMT33///d3rr78+HPdEkd2tEXgWBJpknwXWFroXBJpkX2YmuBh5++237yC/Gtbqatstecj7/fffH144ffHFF3evvvrqPfH/8ssvW8R1m50g8OWXX95xcfvbb7/tRKPzqNEkex4cW0ojcNMIrBHpWt0poCHvnXfe+Quhs0Hzg2g5XTfJnoLu5fo0yV4O+x75ChBgY2Tj++qrr+5v8XELlw2RwMnkgw8+uK/jliN1tKWcwOnI8nrrN29JZr8sp289YTE27fllfY6VOtwrcnd3r7P98nYy8j777LMHPUd9lfHcMbqAM/qha+rJPFhOHYRkSMyoSxsSL+rq7fc1Il2rc+wtMXLeeuuth3lDD37oluFYksVuTsbKA5+ffvrpQSS2cjq2/ptvvrmvw1e8ZU3d559//tDnk08+uZ8D9U2Z1Y7s9yDgjAl0UX983Vv6ldQyTxp/xgfoW/st1Yml47377rsPFzvgyFzxsx4sK47iDC7XcKrtk+wZnblFLSPAxsJG48adG6+bu3XmWZS0o5yYkHnb1U02tchxLGdRQ+r0XwqMrT62cYMw72aOHOoka/VCxiUCuoC1mJFHV/TCbuwnZF6daXsojPAb4ayctTrbHBMjb3SSVYbzsvUk6+Y/2tCpg0glJscAAzC2HJ0gVAkYYktiJe8cfPjhhw/9wD3zyj9XjJ7IH2GRpMp4mScNEWrPljrG4GIFO8USu/Ep8mCZMsnTXt1yjHPZvwc5TbJ7mIUb0GG00bL4WGhu8KRroMwrW2PJjA2E/wdE9lIYjUsZG6ByRn0ZN0k2Ccn2Ob5EZl3NW/4S8dLY6IvN4miMrWnLSEc2TtsTJ4nTfoSzctbqbHNMjLxjSZY+nirRn7l1c8d2iDRJUX0kR/PGYAbOkgnl2TbT9iF2rMSStGSWbc+R1m7m3QsC5VZSy3ym1VuyXqr78ccf70mz3gWQSMHMtDpkXOVm3czpJtmZZ28i3UcbLZsUC+8QydJuFA4RA31G4yrLcdnk0CMD+Wsk2aUT/BqW1L3xxhsPJ+BR2zWc1+oS861p5B1LsltkIxciTkJaIkv841SSlay26HSuNvg6BJeEXkkt85lGB+ZcvZfqmmTHs9UkO8alS8+MQN1oc6OW7CrRoQLt2PRGdfbjlLUU6rijdvSvMirJ0o9NNQk/86RTRs2Pxn2usqWxwYLTWuqpDmtYgkWeXJGfeWSs4bxW5/jHxMh7DpJVB4hVfyOGePN0Rjv8Egw8Hdb8EjmjO/JGc+D4zxlDkIzNCZy0J3r05zT/0UcfPdQxz57Utcd+o7p6u1hidzywXDvJHqp/TlyeU3aT7HOi27IfEGBzYVPyNhnEycImuMG7sT10+iNBuf2I84RZ5VpXy+nnmGvyal2Op57q4lioWYmt5qtNz5lfGxvMwUEbkiwrZtpX7f7000/v54D2tY84Qz7U5S3arHuK/cgdkSybuX9Qo31JBktj1n4Sje1rvbd28ZUcz3L6SUrKyFhCU0fmoJJ4tn9KuurInHqbPOcHHf7+978/nM4h4LQtMVmrS5nYl/3QZY1kJWXH7T98esrMd9+bQ4DFt/Qc5c2B0QY3AjtHACJdukBZq9u5WRdRr0+yF4H99gZtkr29OW+L50VgjUjX6ua1+Pk0b5J9PmxbciDQJBtgdLIR2DkCa0S6Vrdzsy6iXpPsRWDvQRuBRqARaARuAYEm2VuY5baxEWgEGoFG4CIINMleBPYetBFoBBqBRuAWEGiSvYVZbhsbgUagEWgELoJAk+xFYO9BG4FGoBFoBG4BgSbZW5jltrERaAQagUbgIgg0yV4E9h60EWgEGoFG4BYQaJK9hVluGxuBRqARaAQugkCT7EVg70EbgUagEWgEbgGBJtlbmOW2sRFoBBqBRuAiCDTJXgT2HrQRaAQagUbgFhBokr2FWW4bG4FGoBFoBC6CQJPsRWDvQRuBRqARaARuAYEm2VuY5baxEWgEGoFG4CIINMleBPYetBFoBBqBRuAWEGiSvYVZbhsbgUagEWgELoJAk+xFYO9BG4FGoBFoBG4BgSbZW5jltrERaAQagUbgIgg0yV4E9vMO+u2339698sor97+PP/74vML/kIbcL7744llkt9BGoBFoBK4VgSbZSWb2+++/v3vttdceyBRSraRHvkl224SClRcm77333t2vv/56sCNtaGs/5oN5yeA8Pdc85FjnTqs79o1sG42XOCYu33333X1zLgBfffXVe8zefPPNu59++mkkZtdln3zyyYMNH3300d1vv/22qm/aLCbE33zzzd1a3arQK6j88ssvH3DUx/CTn3/++e6tt956WFditgXrGWBpkp1hlu7u7jfzt99++94hl1Rukl1C5nE5G10S61bcaMfPgJwkUzYL5uiDDz54VG77Pcfq7kUD8SF/G9mT/ZAJzsQEsEq8Rv33VqZvSKwQbvrAFn2x/5133rnzwiP7rNVlu2tIQ7LMv1iu2STOW9quydlDXZPsHmZhgw65eS01d0PIevrlCTjJBYf/7LPPHuqzDhnUe1VJfOzmknrsJe1pFIIkmOeUJRks6UofT3j2S0zAi/xoHpZk7qW86kyeORenrXqKwag9sqqPjdrtpawSIHlOXNjwyy+/bFZTbEeEsVa3eYBJGm4lWfYsLkpmvOsxmoom2REqOyyrZDkiPRfsmvpsgm6cpCtpZB31hrXN0zYzxGyUnNDAkzTk+tVXXz2UHbIh50Gs6JMEsmUeDo3z0vU5v6T9YcvWADZrp98cY6vMS7bDHjd70ukrWwmgEnXas1aX7a4lXW8XL90OvqZTLHPXJDuJBx/awDBjtLlLJKMTad30zNNHIhIe68zPGmsbxPrGG288kG21d2QfROpFCfNBGsyRyemGmDCah5G8PZUxv9zVwA6J9dg5X2uPzJlOscyNJIuv+P/Jlm0lWX3h1k+x1ddZK9wV+Pzzzx/dPj4W3yp3j/km2T3OykAnnG/tlEAXF3R2Z+PjZ8iNMNPUm2cBVNKxTjmzxt7mzdvDW7BNfLTd0+t//dd/Pbqt7gWNhGz7Pcf4Tt4eFqc8ra/pTzuJqLabkWCxQSLI28PO+ZbbxfgVmIz+L3atruJ3rfl6YsXn3n///ft9bHRRMisOTbKTzNwWIqgk60ZJOQEZnr7IV+I0X/u5AStnEsgW1aw4YTc/g/YnEVNX2yFndDqr8pW757j6l6Tp6RzdsSuJWHvEi/oawGyEUW2317xEgH7YKQmoLzjx19OcyGqgL/aPCGOtrsq5xvzoIgMsOd1uvUswCy5NspPMlATpKYnYTc3NL+skDRzXckjj008/fehHG2UAQ+ZzPMuz7SSwLaqJTeJCOoOkUUnWcvvVemWAU5Vp3Z7j6itJsOitn9EuA/kRFvjQ66+//oCzuPEoyyxBYlX3SqbYPiLZEYlo81qdba4xzv+T5WI/T/jijI+NLkpmxqNJdubZa90bgUagEWgEdo1Ak+yup6eVawQagUagEZgZgSbZmWevdW8EGoFGoBHYNQJNsruenlauEWgEGoFGYGYEmmRnnr3WvRFoBBqBRmDXCDTJ7np6WrlGoBFoBBqBmRFokl2ZvXxk47leLDDr4x4rsHVVI9AINAKNwB8INMn+8Xyoz8ER12f+Rm9AOpcHbSXZfG5VXen7nGGrbs+pw3PJxjZxPOZlCXnhRf98ZnTt2dvnsuOcctPHjr2oTDx9RjjLxLo+H3lO/Z9LFi+O8HN9S+/bHY3tM7TYnm+NSnnUzfTc8MjOY8p8Vvbdd9999JEF366ln7AHX8tLKZpky0sYRg6zF5I99FrFke5PKbtWkmXzS2Ldaid+wOKnfQ1VBkQzalf77SVffRzC3epv2Jp4Ltl0jMwlGS9d7rz6ggQIcsu80mYLUeiLW17T+NK2n3s833IFNnnRwTjU5QUrZEw7cT+3Li8pr0n2CSSbJxeuwHQSNhO+KcrPKzPrmFycx3Jir/zXJn5tg0I2MlIfx6snr9wMaT/61B1jceJIHUmj9+xBPCo+9e7FyE5xrnWVoCTjxLr22VueuU0/1EfFaUlfbMVO4kMB+TP5EDbld2DJb/nUHT724YcfPnqj0RI24n4NZLJkYy2HQCvJUuYrFcX5Wk74TbJnul0ssbLAJCk3LTYqN1wWlWmcz0VWHbHmlZnE5wZITHnmGYPbLcS5saGTehF7W7CSzzG6VV33nGcB+/ED0pDr1k/dgRevpaQPeIsdc+OpjzRf91EmY8wQsE0/Ie3PsiUb9O28oBz1SYyWZO2tHJ1P+dSd/fCV0W1mSQQf2nLa3RsuT9VnRLLIxJfAa8b/UljDpEn2CSdZNpMkPU9DdUMx/+9///ue9CRDJgYZbGiHgjJGm7YbHUSZgbaQbPbJtrmx0q/mt+qWY+49DRYQ4imfugMf5xg7xfKf//zng0zr1+Zrjxhh2ymfugODvMATX+zPUH0r6/aaxgZI9thP3dGPdzb7nmPWJR8WAKsaKPMEV+uuNT8i2TzJgglk2yfZK/KAQxvAaONgIfk9UqAgn6cZ01nXJHt5p/HELhnm/OTFyEjT6ifO+b/+9a978s07FGwUmR/J21OZF4wSgTiZX9K12jnqRxvwnu0PWfCHentYe9f+DxW/8AQsbkv/lwteW28tK2v2uJIsGHARkqRKG9bbNdxG75Ps4ARXnXhEsm4cbsw4hBu3m691macdPwIyOAWYr+NmPmVkuXJGG7obHhsooeYZ1zrqa94NhX7XFOoJHbtzDsTJ+dT2ikfKSezsn9gqY69x9a/q3+iNPXlqpayujSpnRixyjpIcsWXLp+5sB4ZilP+3m/Jpc+snWfHCvyRV/0jKfGI2W7pJ9g9yGd32ZQNho806/x/OzcM6/v/FW7N1o8l8yqT9119//WiDX3KglFHb1M0/63M8dE0ySWKgT81XG2cijcRglMZW5y4xoa12V5IVI/vlhY19rJsRK/xI/Ue2j0gWTLJf/f806pA12ylWn2FeIVZx8Raw9djHrc1aznrNz/x5SmM9QqrKw4fWTsWOcw0xp1P/j1r7fSSq4lIf8ZnZ/ibZmWevdW8EGoFGoBHYNQJNsruenlauEWgEGoFGYGYEmmR3MnveivM2Ssb1duZOVG41GoFGoBFoBA4g0CR7AKCubgQagUagEWgETkWgSfZU5LpfI9AINAKNQCNwAIEm2QMAdXUj0Ag0Ao1AI3AqAk2yJyKXj2z4WM+Joha78f+0/f+xi/B0RSPQCDQCu0egSXZlipJI/UOkfDaSrjzfxdudeC7u3GEryeYzn+jZxHx4JvIPzeqcjnqPfAGs7VufRx49ZzqSu6cyfJgLRuzaeuFY7a7+lzhTR362wIsRfL7T5zq32OAztPoJz8NmGeX+fI52i9yZ2/isbH0O1ueRxZnnjq/hRRTMVZPsise6sbIwlsJeSNbNy01vTeclW26lHGwkR2wGu1MuTLIf/RNz6pyTGXCtfrz28pO0p/bLOmTw4QDWEYG24E48S3CO3fDzDVBrNtBvy0s4wGLpbVBr8mes8y1OYIMf5Es4IF/KwRm/AbvvvvtuRjP/onOT7F8g+bPgKSRbT5duwDhSfsmlnnio9+qWeMvmTxv6EdTZPGWpS5JLnlwYK+vQl37ZVxv+RGi+lPhoi/k6D4csq+QC3sqgjrRjHJK1h3r0Z64N5PGJQzZUHOxPTB1vN9IXidPHsu0e0+ifBKg9lSCq7vjU1vcRi7skXmVdYx5CXcMQ/Oq7jGfGoUl2ZfbcgJP06iaxtskoOq/o6+bFxra0CbkAlbMUpwzGyg8XIEP59K/5lIkcN1Xi3GTJV9uz7yzpnC/SkKGfpQO7rWE0N2K29Vbr1rFeol36EGl/6TsjPcTQNYLteQLJNYTMmQL+4Iv+SaevrL0m0n5Ln7oTA7BLErf82uNDJCt+axjPhFGT7MpsuUFIPKOmuWlnPZuTGw+xp5y6OZsfjWVdyh2l2bwcKzd4ZVpn7GZXN0jqGZNwLaRa8XK+TvnUnbKUwWZgADfnWLJd8xv77SXGJ0751F3VXxzYIMHJk6y+ONOFmpv9c33qzvV9S6dY/GWNZPWZa/o/6ibZuktE3o1hbbMcbbgszjxNkvfTdy4shzE/Gss62y7FbJC0JZB2s0cm/yeWZJAyaMvPkHKulWTFWYywPedHLNbiipsy00+2zt3aOC9Zh75cZGnDyKYt+iSW1YdGa2WLzEu1ccPPW5valP+fWPUDA0/A1tX/y6UNPpinfttee7xEsuJ9TQTLXDbJrnj0lo1mtHGwEHMTZ1M2XzffzOfmjQw2PcoOBdogh8Di5TRLfwJ1o9ODttV+5t1MaHdtITEXo8RZbJyztB988wKKOtuL3UhmythjGru8EES/6sOUYV8S8cgOcBRLZXjbb4TdSMaeypIcmedzferOPwK6tVMsczsiWXyDrxZdG8Fib5Psyop282Rj8SdhQa5swpYTe6u29svP4NUNPvMpk3G2fgaPTS03eNLqWXVBTwmYWP2xBT2VQ50yViCatgrMtF1S0BgxG5EsbWt7+uXcIXdG7Ko/YFMGfCP9Z2R3xcY+Yj3bJoovnPNTd2AGoeBbt3aKhVx9REd/8JEoLjpqXX3MJ31xpnST7Eyz1bo2Ao1AI9AITIVAk+xU09XKNgKNQCPQCMyEQJPsBLNVb7l5q4W43p6bwJxWsRFoBBqBm0GgSfZmproNbQQagUagEXhpBJpkXxrxHq8RaAQagUbgZhBokr2ZqW5DG4FGoBFoBF4agSbZl0a8jMf/t17q/1XzkY2qw1pdMaGzjUAj0Ag0AgsINMkuAHOoePTwvs/JHuqb9ecgWQgy/xjq2Gc013RYq0s7Zktjl5htxcvnZ+2X853yqM+6WbDBp9H7WP2r7eQNWcezob6YwvoZ4nyG0+c6D+nNRWo+92m/Wq4vzfb88CH7R/X1OVl8jWeFeR6b12+KhbGYjWTNVNYke+JsJcniJKd+U5ZNqJ4ij1WJ/m5sEgGLeWtY02Gtbqv8vbUDmyTWrTbSjp8BOc7dVhn23VtcfTj9e01X2i19zg58klgr7mty91LnvPpmpnwD1JKOYMkLLLygYE0ufZWHtrfykQBIlvUilkv4US7OW9quydlDXZPsibPgJvTjjz/eb9hsIAbrWECEmmfherVGvLRRu8CVuxRXks33FWcd/WuesrVx1uqW9Nlzeb0IMT96u1O1gzn2hGo/8CHMjlPVnzy+mX5d8SCPj4Nd4uAFDGWW25Y618VI3p7KKgGS58SFDWvvLsY3IFnWGiQBhvSTdNNG8LFdll9jeivJsl/Wdz/PjEeT7ImzJ3HmqwgVZZ2bSeZZVG5CtHeR1fQor/was0iTtHNjq6Ra84fGSf3quDPmmRPvOpCGII751B1z6S3VJCBwyjkA55lC+gVpf+lLS/Z4wZEXjLQFn+rr3iJckrWncubazZ50+sqIMKvu3mZeImX871ZOsWBTbxcv3Q6+plMsdjfJ1pWxMZ+bbd2IqMuXrZv/97///ZdTb5JYplGj5pdUyw3SDU+dso7+NX9onK06LOm2t3JJ9pRP3UEanmSdf3FOOyXvUV2221MavzjlU3dpq76XxIpcLz7yHd57sn1JF+YYEjz2U3f1JAtpQNCVmF1b13BLdAnDpXL8htP9559//uj2sZhXrJbkzFDeJHviLEmcP/zww/0CylONdTgSwfxLkCzjuXhJV1Kt+dqefIaUleWzpiUCNr06P+aXbKvY1ZNa9qtts26PaeYZMtSPxcn8ks4VAzD0TkHtwzrI/7+t9XvLY0u9Pay9a7eLsdMTMDZJuoklbfDBW/tIQM5xPbGKE754TRceTbI560ekJU4WIun8/Fnm3azc1Nl8+RFYdHmLDefyFIAMTk22XVMtN3THQxZhNJ51yiS/NM5anf1ni6tNiRG2iKFzpn21HXKcL9sQ5/xn+Z7T6Jx3X/DNaj/2JhFjT223ZLuElUSzZzzUTSIgnyRgPfbwV8ScyAxgkASq7fkXxMjFn66JTLR/S1wxog9YLv3f9RaZe23TJHvizIw2pdyAWEDk+XG7ic2YxcaPBUg5Zfk5u6yjDf2QcyjkWMhlMzSgp/+HyHjcErTeTVM9iR1vrU7ZM8eJmTZrzxLJWi5eSUKJl7eUlTdLzCY3sk39tbESpeX2lUwkF8rBZMZTm8SqbUmm4DIi2Swf9RsRjBhfc5z/J1v9QZzxpWu78GiSvWavbtsagUagEWgELopAk+xF4e/BG4FGoBFoBK4ZgSbZCWa33o7zFhRxvdU5gTmtYiPQCDQCN4NAk+zNTHUb2gg0Ao1AI/DSCDTJvjTiPV4j0Ag0Ao3AzSDQJHszU92GNgKNQCPQCLw0AldLsjw+sPRQ/HODnI961Mc56iM1tO3QCDQCjUAjcJ0I7JJk+WMe/tiHIGEd+wc+5yDZJET/2Kg+I7jmFms6IGf0IoMleSNdxGipz1PLkX8s7k8d86X65x+TbZ0HfVFfyAuoWvfcc/McOKWPpW1bxko8Rz5DWX02covcPbThxRF+tm7pfbupp8/O6ifGPj8Mzq+//vr9M8mzYpL2HpP2Wdl33313+JEFsdmC8zHjXrLt7kn21I1+jeC2As6E17fgbN2QGWNNh1NINnXZasNT2p2K/VPGfIm+FfutdtKOnwE5EkrW4Tf5BjDb7zmuvlp9f013MFhbF2BD/aXuLK3pfqgO3bHPFyTkG6AO9bUebP0QABdj+dk7sVl7TaNyZo99y9WSzeLEqzcT89nt3jXJ1s2wbgSjvG9T4urRq/FRuy0Lvm40uanWuprHMeq46SzVtqwbpUfybadeOKZXzZQR6gkrN0Pa8wYocKKfdYxlmfKIWRyzB/Go+OTbm5ZspI8+pZwRJtY5xpK8PZVjB/5gIM+cH7IBH8dviEdBv+Ud31vW3EjGpcqwSXJEB/L1XcZbdBNbiTr7gE++5zjrrjXNaRafqRcWXsCs4TUjJrsl2dEXO3DyXKiZJ81G6aZQ65b6rU1aJZvciN08GIdQ85SlDnUc9JTUat0oX3XJDRBZNY9svmRBjNMa2EjdTIkraYgf7XV2+15DnHNCmjnl9ZXpH2t25jwkVtln5AtZv8c0vqCf6CNZtqSzfszpAx/kp5y82Ejcl2TtrZx5lABJp69s/UoMdidRVxtdYyMCrm2vJT8iWfyIb/BCvNTje9eCyW5JlsWapIaD1YWaeRc7C7u2zXa17r7xwj91s2QMdap1NX9onKrvggoPxSP5Vi7Jwm5IltiQbesmWvNuAPa9hlhfgFi9pWsZGK8FsPOihLakJRT7IQsfoe1Mgbk/5VN32JkXeIkl2IhPls+CC3MMQeIrzCnEatlWknUNjQgD7K7xhfiH5reSLL4BwYppk+whBM9Q72aPg+Zpry7UzOOwS22zHerV/JLKLChOOLSv/Wpdzdf2dYyqb62v+ZF82yzJQu8mWVH6PfZ05cUSpWvYZm/90rKKO3jPSLDYw1pLshQnbFwLFQP7/eMf/7jHApn1B44zBOaz3h7W3nqrc2QPfoU/jD6MgJxbJFhwqiRL3j8sS1/hInaE3QjrPZft9iTrFTAL0kVZNzHK82ThycSFbt1av7XJqZsvC8PNmbo6nnXKZNyl25AuVnTdEqou2WdJljiIZc2Dn3XIq/kluTn2jGls1qe0O/PiVOeTNtkOOV7YMT/4G5jNGKp/YUe1H3uTiLGz+niVIxa1neV7j/1/QvTELzhx5ZoBp/qpO23yD33qKZb+YOvJzfa3EleSrXb3SbYi8gx5NjId2Q3PzcyFzmLn9hb/F8TCJmQdn5Bbqqv9lkxw4/Tqqm466Gkdt5Q8NUrq1hFL+CzKLCetbUt6UL60eVGHzCUZVZckicQZOTUv9urrnKzpOUtdzl1igv7aXefbcvHI+pRn/dKc7BWj9M20TX1dX7TLkP3w89HpY1aSZc4hVud066fuWK9gWLEAB06wyjOuchPfa0mPTqyjR3WaZK9lxtuORqARaAQagUbgmRHY5e3iZ7a5xTcCjUAj0Ag0Ai+CwE2TbL2V6q0bYm/vvsgslFvdqQfpejvzpXTqcRqBRqARaASehsBNk+zToOvejUAj0Ag0Ao3AOgJNsuv4dG0j0Ag0Ao1AI3AyAk2yJ0PXHRuBRqARaAQagXUEbpJkL/k4QT4WVB/xWKtbn8aubQQagUagEdgjAlORbCVHSak+t3cI6CrnUPtRvWPnHykdowdtK8k6zlqdbTKuz2j2H0olOuO0z3wyf0vzUHvW52TrH8flH9LVuiprj/n06WP1TzzT/5A5+2fdeKmEbyQaPddZ55L1a/vcH/zUXT57ewvPxyY+Pis7+tTdsTin3D2npyVZN7xjiM2JOBfJ5isXjyXGtfZrddqQMZuaL4pwoz8Fl5R5zemKL9glMSzZTjtxpg1y7CdBzYp7XRPYk/69hAnlYDC6UHmKzLXxXrJO3/CtTfkGqK16gIMfCSDNyyj+53/+5x6z9Ket8mZt5xuwsBl/yVdT1hdQiLO4z2ozek9LskksGFIX9CjPG1i8svRKfdRu6VWIOdF1E6obbm5QtS1y6kafstfqsp3pxMKLj1y81Gt3boYSw6hOe7IvZbMH8dEW86M3HFVb6aPf2E+ciU3XfjPk0Z25NpDHL8TJ8hqzfvAp4hqWZHqiq+33lscmyTPJy9AAACAASURBVBHdyNd3GW/RWRySMPCf+orGLbKuoQ2EmiR7Lpz3is2UJMtrEXNDAFwmKskx86TZRN0wat1Sv7VJqwSVm3Ql1ZpHLrok4eVYa3XZzjRYsJAJjOU7lclTbt0of9/pj3+QI0bEuckeq1PK3VO6zj3z9tRP3Um4+CUkDG7pD3uyf0mX9CHS/tJ3Rn31i9Gn7k6VORrnEmWsJUjWr++kr2x97zD+lkStHU2yf55kRzj//e9/f8BezGaNpyNZHJ1NjAWcITdPyjPvRoBj17psV+vuGy/8U4mTMdxYa13NI7LqlMOs1WU702ABJvw8aVHn5m+dsdhhu3ha56Z6rA7qsvfY+T7np+7EOS+awFGc944J+qErFwnYoA9QZnrJBvwE3yEmiC8+f6rMpbFeuhwbIMjn+NRdk+xfSRZi9ctEYr/1YualfeOY8aYjWU6d//d///doM8DgXNw1Xwkj22a69lsDshJnyql1NY/cqlOOtVaX7UznZkhasmch50cSbG9MW36GlHOsDsrYeywhihH6juZnZEfiQ70YsRFATuQN1jHeDAEyTbIUp7RpZEe1M/sp09vDWTeStbcy1nS9Pay9+f+JS3rjV/hZ/UgA7cGibxf/cg/dEs7gswXnJfz3Uj4lyeK8TAwO7CZQ82yInupo7y1UF7p1a/3WJqluzOjhxj0azzpluljRp4a1utqWfG7+jI1t4kJdnrDsLw6eVOxn/lgdlDtDjI3gYiCdebGpc1bbIUdsRzLF0nH2HK/5s3pjTxIx5ayf/O+WlJNp2uJTYDrT6cQ/wEF//KISIzYd+6m7JVmU30Ko/yeLzeLM/1snzvn/2LNiMy3JAnglBjcBNoL6Obusu+Rn8FiU6Jc/N+q1ujUHY/PPDZ20MiWMHI9xCDkem9+nn376IIc6ZayNPWsdmIlJEiz2iFklWcvtl/W1rsqcAafqDxBoBteQ/mNd9uMCL09u1Pk4y2wEi33MK8TqnNdHbrSvlrM3YW9iMZKnXE/7YnqNsY/vaDOxj0SJs74CntdAsMzjVCR7jY7XNjUCjUAj0AhcLwJNstc7t21ZI9AINAKNwIURaJJdmABvReetDdN5m3Ch+1mLvU3n+BnPeFvyrOC0sEagEWgEdoxAk+yOJ6dVawQagUagEZgbgSbZueevtW8EGoFGoBHYMQJNsjuenFatEWgEGoFGYG4EmmSfcf58tIH4UoH/s83Hey6lR4/bCDQCjcAtIjAVyUJWvkSCyfKPk56TROrD9sc4yTlJFrLMP3giveU51ibZ8YzlH5NtwREp9VnY9EXqlblV3lizy5W6nvCtatuaVtpNv/osLDJv7VN3YMXa95lPcKnPg1LmD3+5hjcbrfkIdfU52fpMtfWjz+Adkr3n+ulIltcE8i5RAjH5vZLsc008m/3a6xJz3CbZROP3NBtgEiH+A06HAu3S15BjP2KxTtmHZO6lvl5MQo68yam+kKLqCwb51/aJ7aky6xiXzOsbvhjBNxOt6YTdvMDCN1uxXj/88MP7F1P40gVwurUAibJGxDLtB9dcP9d00TEVyeLwvMmJH45s2o0vr8S5SrS8LvbMk2ZT5G1HXlnaj0m3zDg3lFrvwskTTz0RIJuxkIPMlJdOt5ZGhjraLnXJTZ5yX3Bex0Nf6rOvNlQsU6Zjzhg7N9ppfss80Mf5tF+dB/IzYoXe+IGBPP4iTpbXmHb8DK4n4iWZs7zdCBvyCzrk67uMtTtjfAOSBU8IBQx98b11h3BNedeSXiNZbaQN66dJVkReOHZBQxr+cFacGVLw/cSoxYJg46SedL5fNfO2c2OsV/DZds1c+nG6ZBEZRn2xITcvdM9Nyr5L8WgccbFP5pEvMVCf44FN6kJeHJRlTD/qZw85J849vpT+sWYj+INn4pbtwX4Jw2y3t3T6BWl/h3yz+gztwYfXCZ4qcy/YMNdP+dQdpzNuGSdpSLL4j79ZLjqeOi/eDtZub6Gn3CbZROMCacmDzRHHJWaRs5iNUy0XeW6s1Gc+07VulE/56KPDENfTUJVNX/qgl6HmLR/FLNB6m5gysEg9SDuGGCgvcaobpG2I0R17Ui66zh6cE4jVizLL2FTXAnh5wSLZVkzIz0qy3BlCd22qvrOEDe30E+7SuDYpP1Xm0lgvWS7J4iusBW7/Wuat4JE+Ein2c5KFbO1f2+NT/J91fcdxbXdtedYcp/v6juIm2QvPNE6LU2YgbzlxBvJsGHUTzXym6Xsor3wWm5s0ZeTr/2FVWbRDn9Sz5pU/irHVDdD6EfFaRywGlokXedJLhEC/1LPKUd5ssRcleUE0mruRXRWDEX7MzxKmI5l7KUNviBKbCOJkfqueYOkdHWV6UjtV5taxz92O9VtvDzvna7czwcATsFhy+3iEJWPkLelz27Bnef7/dv4fbZPshWeMTa46KnnKcexKeuZxZDZV+9LeE0klwppf2hiQlRs1MjMPVFUWZWw8tDXUvOU1RpYnhFqHvKWNnTrGIFRbsGHUz3b2A1vwMl/Hny1fMQejnBPtr/NZ2yGn4jcqmwEf5jgvEqt/YwO2JRFXu/BRSMl1tiRz7RRYZV46LxGgB34BWYKDAVu5JZxf4cFufMfTqbh4sWFfYv1ljbSz/bWkK0ba1SQrEheK2eRcwKpA3o2OtLet6mbgBkE5t7C87VqJsOYZJ+W68boRO17eJkMG7awjltTRIzf0mteuGtMn5ZFe0oU6car9GM+Q2FlmXG3Gvuxru1njxCXnA3ucW/HVRsudh6wHG8uNq1zl7DWuc44fZ9BGfYu69HV8XGKxH219lAW8ZiJYbJBYndMkU+q1b6m89qvy2LtuhWAhUH2h+krWidno/2z1q5niqf66eCZgW9dGoBFoBBqBRqBJtn2gEWgEGoFGoBF4JgSaZJ8J2GPFeivOWyUZz3bb8Vjbu30j0Ag0AteKQJPstc5s29UINAKNQCNwcQSaZC8+Ba1AI9AINAKNwLUi0CR7rTPbdjUCjUAj0AhcHIEpSZZnrC793GZ9BvDiM9kKNAKNQCPQCOwOgXuS9fm//AMb0vkc4HNozh/75JhbxzgnySor/9CIdD4LONLrpUl2pCf4PWc4dX6eU6dzyc4/NPM56zXZrpHqJ/ZNebTxueg1mXurSx87RX/Wcn3+EZm39qm7+iws/pDPw14DJqf6rs/D1s/ZWe76qn506nh76PdAsiwQfjxczo80jlEfSD+n0nvcxCFXbD8UWCj5hpxD7Z9a/9Ljoe8e5+epONKfOZYcn2Jn4pPpc+j40jJY5/mRhGP9DfvBNGU8VeZLYzAaz3n11X/5BqhRe8ok2dGFOnV+9o624nYLL6QAO/bWkc2QLHXivITtjOX3JMtiYIHwImwcgx/pJFkA8CojT7g6ocZnnj68XYmrEq/mcDIWsGXKJKYvodbnhph6pBNrA28mUqby1O1QrAxiQ9XFMSnnrVHYV8fDRurAUDu1AdngZ5/E0jFHMeMtkTo6gcsIG3RhbMdTD8Y4dX5G+s1SJh7Oo/mt86CdzGMSSvq9bWaKq/7k8RlxWrNF3/zhhx9WMVHm6PWCa/IvVccc53uFydd3GY90w6eW3lVc24Ndvue41l9jHkJlH8oLi5shWRbJ3/72t/vfjz/++PDqQTdwJ5zF4mY9Wpy0JxBDMjiSm1ku2tpX+TVGTvYbyZK81MuFT/nWUMehL/KUkXnkY5u25njqpy7m0wZ1AgN+h4LjSZa5ASK35hmbV9gRp3z0VWfic8zPId33VM8cSo6kIVcuhizbqmv1XfI5N2K8Vd6l26GvfkLan2VL+qVvJ7a0P1Xm0lgvXc6akwBJp6+svR4STCDZ9IelCwv96BpPcEvztUSyvnIR3K7llYpg8OgkyyJh0vnhKJzG/vnPfz6QraAloegk1mU+Fxn1NZ9t7U/s5pdOSltDLmzL6gKvedstxRAV+mWQvFIPSSkxoI94UT7ST7nWpcw6rm0zruNlHXpK6FkOBpQTG7JtnY+aX5ofZc0Y6xcQa35A4hiSVQZzMgrUsyGnz47a7amMuT/ls3TYqJ0Vl1Nl7gUX5heSxVeYT4jVsjWSrfqz5kafs6Ock/ExsqrsGfMjkk078CNwqZ/ByzYzpe9JFseBUCEAg6RxCZJlcfIzkHYhUyZR4aSGusBr3najGHmMQZ8MyE89sg7M2JjtI16Uj/SzL3YkIZJfGsM+xHW8rEPPlGkdujXJisbvsXPDpuncrWH7uPfvOebr0JxRnz47krOnMnTlws81JU7mR7qCHzjmBaNp7SfvKW6LzNE4lyrDvnp72LWWtzoP6YecvO1Me+TcIsFi+yGSpY3/930NJ/yDJMsGVDeVzCdp0DZvobrQdMKa12FZfAYXohuUMs3Tzja5AeDIeRqpeeWPYmSnfNs4do6TdUmyactIP/sldujIJkXZoYAuOV62z7GzXD20rebrfNT8ktwcY8Y0eCTmpDMvTknE2sk8eAK2rMZb2tQ+l85X/2Luq/3glkRcda5rbknmTCc3N3tsxS+2fOqu4gJuXOxKzOTBdiYcqk1PyR8iWfwGfOoXnZ4y5iX7rpIsjsFic9PxKjVPTRIFdQDDrRU3LGIcylDzVa5tGdOxkOln1mp729C+LvCaV4caM6FcGCjLeKQLddpe+6EnYxLUE71qyH6Myy068aptM0+/Y0mW/jk/6J9j1fmoee2omKRes6axVbsSE+zR7pxT7aRtbU8d/qI8/0vBPrPEdd3pz+qvjSO/ps1ozdHW/2sDz9mIBV/I/19d+qRdltc+SbBgxAlWXzHO/uJ9bTHkqi9ot//3mnWsn2shWObwnmSvbTLbnkagEWgEGoFGYA8INMnuYRZah0agEWgEGoGrRODqSbbeLvU2BfGebut5Ky71Mz26PXmV3thGNQKNQCNwZQhcPcle2Xy1OY1AI9AINAITIdAkO9FktaqNQCPQCDQCcyHQJDvXfLW2jUAj0Ag0AhMhcHUk6yMyPoJziblYe9zmEvr0mI1AI9AINAKXQeAgyfrMYP7xDenRM4TnNAGSzDG3yj4nyeZzg/4R0pY/lnppkgUn9SM+Bbet+F5LO+f2WD/OP1DzmWkxyXmYcQ5cO/jQFj/HbvcH/S+fcVyrE7MZYl5I4fOdPte5pnd9ThZs1p6V9Y1YazKvva4+P3wtr1Rk3jaRLBsGP4DgRxqnIf1c4VSSfS59kItO/A6FS5CsejEnEMfSCwMO6X4L9WCF/3799ddHXSyCaRIrciTTTIMh5c7JDJjiN/nGtFN9uOKQtq/VZbs9pdXZ1/vlG6CW9JRkR2uw1oF7feXiktxrLRcTsAZn89dy8XGQZHECNhbe5ITT8COdJMuG4pVsngzqRpN50qd8Bo+JcCxi8obUIx1cG57yGTw2nfp+57zyRxfHtO3oM3hrdegJftqXWGrjKE5ccVDmZgmXJImqf9ZhC3KXMB3pMVsZNm7FWFydY/P0/9e//vWIoJzHxHPv2OAvzLXBdaa9lh+K0xdr27W62nYPeeYxCZB8fZfxSE9JYoQdZfiFr1gkzyn5Ft74NMKKMnDlrVr5NjDeAIUPenGz1HeG8s0ke8pn8Oqiyjxpb0m5YaVT1kU/ApPJwWGJDSNZ1LMZuulBLkuvKFROxsiEYOlnqGNnXvLCRkKOZx321br7gviHNraL4r8kE1fk57t1q4yaT2HIcQ6I88KBvPhln5nT2LSVZJlfT3r6ExeblP33f//3gz+Jv3W0nSGkD5H2t8X/9Gn8pfoIdXyBZlS3d1zQHZJl8yeNrzivSQjVDknWi2ViT2XgCbaQB2nw8tWq10AoFYstefESI9YMFzPitEXGnttsJlkMd4OWdA59oQeQcpFmPtMAVPM6YwXPzV8Hlqhth244Lu0M6O4GSVnN224p1u6sr3qgj7qwIJPExYvyWpcy1V3biMHlUKCNfdSBPiN5KRMc2DjsS+x8YV/dMA/pMVs9Nh5LsmyyXsToR5Isdcpbm+c94oQPsdkz5/pAXZNb9Kbvkt+s1W2R/dJtmENI1nmVbCXerfrgZ37qDgzA1Z9kS/5WSRYcwdqLMdbQf/7nf9774TVgcpBkMb7eJpU0XppkJQWcluAmh44GicU2o3ajfvavMW1ZAMjNgHzKRwF9TiHZugm5IEdjZBl60JZA2o3eeUp8ar+0IeVg39JmmTJmTmOjWB2yQ7/K9s4zt4spT7xmww//4SILvQnaa/4QPtazXsCBuIa1utp2D3n0rbeHnVdv927REznedqZ/vT285f95t4xzTW3A5Fjf26v9TyJZNhk25rpRm886AMuTEnUSA+DUvM6c5MZ4niLoQ/88uVE22hxw8lNPsug1mmx0YeyluiTZtMWNGZ1qSLyoZ+Om7FBI7Kpe1OXmryxxcg7sZz51ts+1xdiYpKl9YlPrwCbng7R5YrGzv3nl7jmufjnCBnuSiEf20Gbkb7RdqxvJ2kNZEiDz+tRP3SXhYh+442fX9NWZp87boU/hPVX+S/c/mWRZSCxENxQWH79cYG7clnM7yo0nNyWMrvkqN/s5Fids/6+0trcNOp5KsvRVTsaUE2q9tqfd9MvNum5mOeHZDwL3/2qyzShdscvNbITLSH909JOC2qY9ozFnLqvzxhylrWKW86a9YK0vkDbYxzr91foZ4sRlZDs2YZ/+g03ps9QlJtR5C7DWzYAHOjKv5/zUnZiJC+u8Cfb3D7kf85jULP6DngdJdiZjWtdGoBFoBBqBRmBPCDTJ7mk2WpdGoBFoBBqBq0LgpknW//f0Fl/G9f96Lznr3qZL/Uzn7blL6thjNwKNQCPQCPwVgZsm2b/C0SWNQCPQCDQCjcD5EGiSPR+WLakRaAQagUagEXiEQJPsIzg60wg0Ao1AI9AInA+BJtnzYdmSGoFGoBFoBBqBRwg0yT6CozONQCPQCDQCjcD5EGiSPR+WLakRaAQagUagEXiEQJPsIzg60wg0Ao1AI9AInA+BJtnzYdmSGoFGoBFoBBqBRwg0yT6CozONQCPQCDQCjcD5EGiSPR+WLakRaAQagUagEXiEQJPsIzg60wg0Ao1AI9AInA+BJtnzYdmSGoFGoBFoBBqBRwg0yT6CozONQCPQCDQCjcD5EGiSPR+WLakRaAQagUagEXiEQJPsIzg60wg0Ao1AI9AInA+BJtnzYdmSGoFGoBFoBBqBRwg0yT6CozONQCPQCDQCjcD5EGiSPR+WLakRaAQagUagEXiEQJPsIzjOm/n222/vXnnllTviDo1AI9AINAK3h0CT7DPOeZPsM4L7RNFffPHF/QUQF0Hvvffe3a+//npQIm1oSx9+r7322t3333//0E+ZW+U9dNxJAluwaWTbkoraLCb2/e677x51+fjjj+9l1/JHjXaa+eSTT+5effXVe1w++uiju99++21VU9a97ROXb7755r5frd8ic3XAHVWyRt5///0H+z///PO/4IWfvf7663dp9zVj0iS7IwdtVV4GARZ0EiFEAQkcCrTjZ0CO/Yj5UZ+ybbv3+Oeff757++23Hy4a2AjJU35MGPUTk5R/jMxLtkV35lVihXDTB7boBobvvPPOHRcYpCGhn3766b4rpPThhx/e122Rtfc2X3755T0+4IUvvPnmm49sE4sPPvjgEa5pl21mvCBLO0w3yYrEkTEL77PPPru/OneD5aqVjTdPPPW0wzA4nycG+rhoKcf5kOsVsHX0YxzLcV6c8VCoY0kAlL/xxhsPmypyknzShmPHPKTTJeu1C1sJ5rfgSR/n0345P8gjL8aXtPPYsdEb/zKQZ97FyfJDMTISE/wMcv3hhx8ekfghOXuor5s9+bfeeut+fn/55ZfNKootxIPfQLLgRB58kSnpbhY6QUNt9QSPyl6kJCbVFHyGi5JrwaRJts7wxjyLhI3ZjVenyQ2GRVmv3iW90eZlnTLcoJDDePwMtDllM0eGY5N2LORmnjQ/wsgO9ZgtTltIM4dfffXVX+ZpyS7naImATp2XpfFeqnw091m2RY/0V9p7IYK/Je5bZO2hDfa42ZNOX9lKANjtKTZt8hY0a/gYwk4Ze08nfuiKH3CBgb2cePEv7xBQLyasrSTmvdt5SL8m2UMILdS7AeXGYpldRhsLmzDtRiFlZT2bFSdc6g1Lba03lkg8jRJLrMhALvJpx4InJqCjeipDclb2jDG2cOEDsXqStyzxHdmG/Z5kaUtaLG0/M8lyBwUf0Kbqz9q4FNf2yFHWVoyXZF+inDmGIPEVCBZitWwryWI/uEgmnu4sg1iUfQkbn2tM5psTumRJPm+Tj0hWXWpfy2eNm2RPnDk3FBYdmzaOYZkiKasnWRedbTJOWVn+FJJFJ36G1DHlQiDZjvyImJUza4zNEAkbG/NDWMK92pjYUQdGyEKmgfmtZdbtOUZv5hubCOJk/pDutEuycKNMHzKdfnZI7iXrtYH59LTpnJtf0w+/ApP8v0XKPB3TF5whn604r423lzpxk2DRC1Id/TEYF6qJjzbk/+1aNmvcJHvizLnh5gZtmSJxtkqytPcEZTvjlGWZMbJzc6p522XsRskGSkB+PX2xuKn/9NNPH07K9Ksn55Q7exp717AUtyRibK6YI6cS6qhsBryq70ma+LAB25KILRcv6pfCaC0std1Tuf+HiE7YCSGmneAEefBXtDXQF5/xFEs9OCfxjgipypkpj3385XAS7Ej/PsmOUOmyRwiweFhsuTlZxsJhIXnlTuxtRoSwMLPORZuyHg0WJwv71c29tjefY6ETZOp4tFFXdM+Q/Rwz+2XbGdPYq13Vdkmjkqzl9st6sLHcuMrdO04552mbemsj7TKQH7XPNrOSrMTqnFYyxfYRyVYyTSzssyQz286Wzv9X1b5333334U6A9lSSzX5Lp1v7zhb3SXa2GXsBfSWT3Ey3bKQvoFoP0Qg0Ao3AVAg0yU41XS+jrKfbJFlOMVtPzy+jZY/SCDQCjcD+EWiS3f8cXURDCNbbPcSHbgdeRMketBFoBBqBnSPQJLvzCWr1GoFGoBFoBOZFoEl23rlrzRuBRqARaAR2jkCT7M4nqNVrBBqBRqARmBeBJtl55641bwQagUagEdg5Ak2yO5+gVu95EPCZT/6oa+tfTftok38Qls8+o6Uyt8p7HstOl8qzndiEfdW2Q1K1nb6j54Mpm/X5x3yGMz/PdggT/njQtxzhE74lKstnxWTJdp8r1u7+1N3dXZPskrd0+dUiwCaXRAhBjIihAkA7fgbk2I+YH/Up27Z7j+vLItZejFJtwe41m8Wkvv2sytljHt2xz7c2QbjpA0s604a/yK/vOK4vqSCfr1lckjdLeb4OsdqKDfgZ9vJGucQ17bPN6HWL2W6WdJPsiTOFgyx96g6R1HviIWZDNuB8nhioc9FSjvP1p+5E6vyxp1Hnw/yWR5To4wnPfs6dmpJfIxzb7S1Gb3zWQL76rXUZsyFiL/Eo4NOQ6y196g7fWPpGrBdmkvZa2xGeM5VhG6+hzFcsepGiv4lD2oXPXNOFR5Nszu4RaTYkNmY3Xp2GuAbJE6cjzUbtJp9trVMGeTYoNjDGq5vgKZs5MhybtGOhR+ZJ8yMw/oynkMTWdNpCmjnsT92N5z79QfxqjC/hh1wcelGpT+Hv1NEmca8y9ppn/bnZk05fqSfUtMF+vMLU26beZrbO/mBDmySilDVzemQrpNufupt5Vl9QdzcgHCmJ0A2G2E2H2JMS5ZJXVTdlZR2bVX1h/1Lb7EdaIkld1BEZyEU+7fJEgo7qqQzJuY4xU97Nvj9193jWmOtTPnWHT+Bb+ob44lv4mb6W5Y9H3m8OGyDZYz91Rz9eku97jllf+aUdMJF8WX/8ruXWqLPJfPen7n5Ho0+yesWRMZsSi4UFVUmWsvzSTrahj+RVh8x2WfcUkmWsHE+9kZ9y2SSznZun5OxmmXrNmMZmLia86MGGJdyrfYkddWCELGQawKmWWbfnGL2TLMVJ8lzSvWJgv3/84x/3GOs/GaefLcndQ7lEwXzmHy1lfqQn/uQJ2Hpvk5o3ZgwI2JOt5TPH4panc/6v1guL9IWlP/zK/9udGQt0b5I9cQbdcHODtoyNJzdxys3TPgk4h09ZWU4aGfwMNW95xm54EiTycWrztEVX8v2puz+xFTfnTEwr5uBWCXVUZv89x9X3qg+jO7YlEVPGhpr/lVDlaHNtZ/ne4yRH/AJCBAcDOEEenloptx11BGyHdOtp1XYpT7mzxsx/f+ru8ew1yT7GY3OODZfFkZuKZW7SXrFBYGzGLDYCi886YhdZyqqKVJl1c6/tzedYkEZ/6u53ZJgr54B0BrGuJGu5/bKeObTcuMrNMfaYrr6iv6qrNkoelme/pZPJrCTLnEOszmmSKfZjeyVZyiUb+3mqq/IsF8vZYy5K6om1P3U3+6y2/mdHQDLJzZR0ksrZB22BjUAj0AhcIQJ9kr3CSX2qSZw6INQkWU4xW0/PTx2/+zcCjUAjcC0INMley0ye2Q4I1ltdxH2KPTPALa4RaARuAoEm2ZuY5jayEWgEGoFG4BIINMleAvUesxFoBBqBRuAmEGiSvYlpbiMbgUagEWgELoFAk+wG1P3/yfxDoLVua4/irPU7po4/RJrtEZFj7Ou2jUAj0AhcAwJNshtm8dwki7yn/qVuk+yGiVtp4jOf/FHX1rnw0Sb/IIxnQrmgMihzqzz77SXGFmzCvmrbko7aLCb2rS9e4IJw6RnaJdl7Kc9nP30H8ZpurO/6rCi48EzsWt2azFnqfA5Y+/tTd/3Gp2fx3UMn2SbZZ4F9s9CK/9YLFtrxMyDHuwnE/KifkWTryyIO+bAY1HjUT0zyzVC1317z6M68+rWYfAPUVp3BdvTGJ/qv1W2Vv6d2+TpEfIGnEvKCS3t5X3PimjbYJvtl/WzpmzjJMplsiBlwgHw5Ps7g1bgbaZ5cRlf2yLSPMX3daHi7kuW0xXlyHOvQz0B/y+tjMzMbIgAACB5JREFUM1lHm+xn/xqji6cT+kgAlNfXO6Kj9Wn7kj51rBny2qU/mK9Yj2yhj35gP33F9uTF0LIZYvROf9LXxGmrDchITFwLt/Spu4qV2ErUWb9Wl+1mTLNGeFtWvtXKi5Q1u/GZ+u7nGe1X55sgWSaUHyTHBkjM5sGGgCNAtkwsoeYpo329Cq8kpdPQnjo2YzctxsqNt+bvB/7jFW32oUwdSSM/ZeR49t8SIx+5BNLIMWSeND/CyH77zBanLaQh1/7U3WNfcO7TH7bMM37vxzJo74UI/pa4b5G1hzbY42ZPOn1l6wv9sftWTrF1zhI/6vADSLc/dVeRuoK8ZMWkc7okD7nwoyxPep7aaGMYbRD0y5MgsiRB6nKzqXlk29YxiNnUHN+Ydixo4tSJ8STBlFHTEonyiOlLQK88zTMG7QlutKSVkePfN5rwH+eyP3X3ePKY71M+dZdSkKFvUU7avLjjc7MEdIUgj/3UXdqH/eBya6dY5rs/dfe7J9zESZbFArmyWCCKv/3tb/eOTzqJJhdHpkcbhFfpklfebkTmqSQ7IjLHyjoXb+o5SidZUp8bYZ7akU2dgby2JTFbP2sslmvztWRbYkcbMKoXS8xLLVuSt6dy9Gae9TFxMn9IV9qBqSc8N9n0IdPpZ4fkXrJeG5jPYz51p87sA2Ay+r/FtTr7zxqLW94m7k/dzTqbG/Vm0jmxQbSkWeTeIibPQmCTWQq0qbeL2VSWNotDJFvrHRcdcvO3nJixHI+x2bDMZ7tMu1FqG+NyajdPW2SRBxvqCUm+Ke9a0tib2CW22s/GWueitkNOJdRR2Qy4VZ+UNPF9A7YlEVte/czyjEdrKOv3mvb/ENEPO7d86k5b6IvPjE6xa3X2nzHGj/pTd49n7iZOshKpGyubRW6gko9X2tbZz3Li+ocvWWe/umHVPFOALvZVr1pOPboSUhc29q+//voRUdw3GvwjISML/SBTZabc1IHy7Kee2W8w1FRFS/hjhKThfGqY5eKR9WBjuXHFVDl7jXPO0zb11UbaZSA/ap9tZiVZidU5PeZTd2Bya6dYLh58fEfM+lN3uRI6vRkBNpY8xbgBXwMRaUtupls20s3gdcNGoBFoBG4EgZs4yT7HXEKmSbKeNJOYnmPcl5A5sqXa+xJ69BiNQCPQCMyOQJPsiTPoac9bIsTXcIoVDi4W0rZDtwPt13Ej0Ag0Ao3Anwg0yf6JRacagUagEWgEGoGzItAke1Y4W1gj0Ag0Ao1AI/AnAk2yf2LRqUagEWgEGoFG4KwINMmeFc7DwvJxofzDqcM9u0Uj0Ag0Ao3AbAg0yV5oxvjDoibZC4H/xyv//MOuLfMw+kM3+te+XkTN9owsM6Hu2OXz4IdmiD/2E0dj+tbnQ8FjVH5I/h7q89nPp37qLu0BO54pzTcjZf2MadYJL+zwWdn+1F1/6u5iftwkezHo//I6RDa7U0ix9vOFC7xN7BR5l0Pkrx+BGL1AZYt+o37gxMVIfWvaFnmXbuMc+9YmCJeyYwJ+UT8SwPqnDEyuiWT7U3d/9Yw+yZYreK7Gc4Mk7RU6MYuDQDkvVOfqnLTtqGejYaOl3r51YS6RLO3sUx+bcQzqt5408nRCP09elOcHDrApdRqd3Ko+f3Wn/Zdol/No/ljbJFRwNDA/zB8/0jOFqrN+KE5bbRED24MPRNKfuvvz9YqS7v/+7//en/qOxVhs9x57qs2LCC9S9DcvXtIWfIYLEN+DnXUzpptk/yDMLY7O5EOeOA+bCRsz/SA8nYaYdpIvTuFGw+Iy0E/Cy7LcnGljPse2/Skx8rSVNPoaMk+aH2FEKPaZLU5bSDOHx3zqTnudb/M5n7XONnuOR3OfZVt0r37uBQzYJO5bZO2hDfa42ZNOX9lKANhdT7ESjSTketyDzefUIfFDLnb2p+7OifBEstgUOeVJKqm6dfV06QaUG8uoDFksJj9IoOzclC2jv+MYS8QSwtYTrDLtpzxibCKguxcNtGMsYgK68CMo4xo2A2zhZHXKp+7uwRhcdFTsZiXZ/tSdM/x7LEmc81N3rCHWFSe4ayZZ1kR/6u53P+qTbKwrFgAkJLGxyPKW6iFCZfGwwWY7xB9DsoeIDFnoh56H2jJ2kqV5STb1cvELh1hIzvaxftZY/PL2cJ2vQ7ZVTMFGnDI+9oLo0LjPWa8N+pQ4mT80Nu3A1BOem2ziYRr8ZgjawHo7x6fuwJSTnDhknGPMgM2ajuKWt4n7U3driN1YHQ6CwxO7cZAmsDm4OZNmY8oNelRGP+Qgk0VmGJUhT/m2W4ppy28tuFHaDl29tW0/9KC+P3X358YvbqO5AMO88BLHjMFzFiJR7/RjyqrvU4ZdEAN1GcRLP8s606yhGf/wyVu72IGd5/rUXcqreIrZjDF+1J+6ezxzN3+SdYPIq0qdvtZBRBLwiFCzDDJTZm7WyLbcOAkYGZYTu3HVftnn8ZQ+zmU/9OhP3f2OT+JMOoPznvNmPW1re+uMZyRZdK++4sVl2oVP0i4D+RFW2WZWkpVYXZPn+NSduCi74mn9jDEXJT6+I2b9qbsZZ3LnOtdTwc7V/Yt6kkwu/i0b6V8EdUEj0Ag0AjeOwM2fZJ9j/mcnWU4dnEySZDmdbT09PwemLbMRaAQagRkRaJJ9hlmbnWSBBIL1dg/xoduBzwBji2wEGoFGYHoEmmSnn8I2oBFoBBqBRmCvCDTJ7nVmWq9GoBFoBBqB6RFokp1+CtuARqARaAQagb0i0CS715lpvRqBRqARaASmR+D/AY9KotUmO3aPAAAAAElFTkSuQmCC)\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fAUFfhaPu9UA"
      },
      "source": [
        "**Experimental Summary**\n",
        "\n",
        "The experimental data produced a validation accuracy of 74.23% and a loss of 1.164. This model appeared to plateau in accuracy and loss at around 25 epochs, but this may reflect a local minimum. With the experimental data, the model produced a test accuracy of 78.19% and a loss of 0.8606. For the eland, the model produced a precision of 0.84 for the side view, 0.60 for the rear view and 0.78 for the front view. For the greater kudu bull, the model produced a precision of 0.85 for the side view, 0.82 for the rear view and 0.86 for the front view. For the mountain zebra, the model produced a precision of 0.81 for the side view, 0.75 for the front and 1.00 for the rear. \n",
        "\n",
        "**Images of the Eland**\n",
        "\n",
        "The confusion matrix revealed that the algorithm correctly identified the side view of the animal more frequently than the front and rear views. The model correctly identified approximately 82% of the photos for the side view of the of the eland, with 9% being identified as front views of the eland, 4% as rear views of the eland and 5% being identified as a bull kudu. The Model correctly identified 67% of the images of the front view of the eland correctly, with 33% being misidentified as the side view. The rear view of the eland fared even worse with only 58% of the images being correctly identified. 25% of these images were misidentified as the rear view and 17% were misidentified as the side view. \n",
        "\n",
        "**Images of the Bull Kudu**\n",
        "\n",
        "The model correctly identified 89% of the photos of the side view of the bull kudu correctly. 5% were misclassified as the side view of an eland, 4% were misclassified as the front view and 2% were misclassified as the side view of a mountain zebra. 64% of the images of the front view of the bull kudu were correctly identified. In contrast, 14% misclassified as the side view of an eland, 14% were misclassified as the side view of the bull kudu and 7% were misclassified as the rear view of the bull kudu. 60% of the images for the rear view of the bull kudu were identified correctly. 30% of these images were misclassified as the side view of the bull kudu and 10% classified as the side view of the eland. \n",
        "\n",
        "\n",
        "**Images of the Mountain Zebra**\n",
        "\n",
        "The model correctly identified 90% of the images for the side view of the mountain zebra. 2% were misclassified as the front view of the zebra, 2% were misclassified as the side view of a bull kudu and 6% were misclassified as the side view of an eland. The algorithm particularly struggled with the front and side views of the mountain zebra, achieving an accuracy of 40% for the front and only 33% for the rear. 33% of the images for the front view of the zebra were misclassified as the front view of the zebra while 27% were surprisingly misclassified as the side view of the eland. 56% of the rear view of the zebra were misclassified as the side view of the zebra while 11% were misclassified as the front view. \n",
        "\n",
        "**Conclusions and Discussion**\n",
        "\n",
        "Overall, the algorithm proved to be much more capable of identifying different animals than predicted, and creating different categories for the different angles of the animal actually decreased the accuracy of the model, with images frequently being misclassified as the wrong angle of the animal. This was particularly true of side views, which never achieved an accuracy above 67% (eland front view) and were as low as 40% (mountain zebra front view) and 33% (mountain zebra rear view). When the model misclassified an image as the wrong species, it was usually as the side view of an eland. This included 27% of the images of the front view of a zebra, 14% of the images of the side view  of the bull kudu and 10% of the images of the front view of the bull kudu were misclassified as the side view of an eland. \n",
        "\n",
        "One of the issues that this experiment had was that the data was very unbalanced and this may have impacted the results. Originally image augmentation was supposed to be used with the model. It would have a rotation range of 40, a width shift of 0.25, a height shift of 0.25, flipped images and zoomed in images. However, this code actually produced a lower accuracy with the control data leading to the assumption that it would not improve the accuracy of the test. However, this failed to consider that the data was highly unbalanced in the control group. In hindsight, the purpose of the experiment was to see if the algorithm improves with the different angles, not to generate the most accuracy algorithm possible. It is highly probable that the small sample sizes for the rear and side views may have contributed to their failure in the algorithm.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zB3sfRH7yvy1"
      },
      "source": [
        "**The confusion matrix for the ResNet101 model for the control group.**\n",
        "\n",
        "![image.png](data:image/png;base64,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)\n",
        "\n",
        "\n",
        "**The confusion matrix for the ResNet101 model for the experimental group.**\n",
        "\n",
        "![image.png](data:image/png;base64,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)\n",
        "\n",
        "The ResNet101 architecture benefits from additional hidden layers and can potentially perform better on more complex datasets (Mamun 2019). However, this model was not able to produce better results, with the control model producing an accuracy of 0.6382 and a loss of 0.7943 and the experimental model producing an accuracy of 0.325 and a loss of 1.943. Control algorithm classified 51% of the images with the eland correctly, with 7% classified as kudu bulls and 42% were classified as a mountain zebra. 46% of kudu bulls were classified correctly by the algorithm, with 40% classified as elands and 14% were classified as mountain zebras. The only animal with a high success rate was the mountain zebra with 90% being classified correctly. Only 1% were classified as a bull kudu and 9% were classified as an eland. \n",
        "\n",
        "The experimental algorithm struggled with the data. Side views of the eland were classified correctly 54% of the time, with 42% classified as a front view and 4% classified as a side view of a mountain zebra. The front view of the eland was classified correctly 72% of the time, with 28% classified a front view. Rear views of the eland were not classified correctly at all, with 50% classified as a side view, 42% classified as a front view and 8% classified as the side view of a mountain zebra. Only 25% of the photos of the side view of the bull kudu were classified correctly with 2% classified as a front view of a kudu and 74% classified as side or front views of the eland. 21% of the front views of the kudu were classified correctly with 79% being classified as an eland. No images of the rear view of the bull kudu were classified correctly, with 70% being confused with the front view of an eland and 30% being confused with the side view of the eland. Only 49% of the side views of the mountain zebra were classified correctly, with 41% classified as the side view of an eland, 8% as a front view of the eland and 2% as the front view of a bull kudu. The algorithm struggled with the side and rear views of the zebra with no images classified correctly. 86% of the images for the front view of the mountain zebra were classified as an as the side or front view of an eland, while 67% of the rear views of the mountain zebra were classified as the front or side view of an eland. 33% of the rear views of the zebra were classified as a side view. However, the algorithm only ran for 30 epochs, and with additional epochs the algorithm may have performed better.\n",
        "\n",
        " "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FnJ3e8Mg8wQ4"
      },
      "source": [
        "**Bibliography**\n",
        "\n",
        "Mamun, Iftekher (April 7th, 2019) *A Simple CNN: Multi Image Classifier*. [Towards Data Science] Retrieved from: https://towardsdatascience.com/a-simple-cnn-multi-image-classifier-31c463324fa\n",
        "\n",
        "Chollet, Francois (2018) *Deep Learning with Python*. Manning Publications, Shelter Island, NY.\n",
        "\n",
        "Shah, Tarang (December 6th, 2017) About Train, Validation and Test Sets in Machine Learning [Towards Data Science] Retrieved from: https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7\n",
        "\n",
        "Labeled Information Library of Alexandria (LILA BC) (n.d.) Snapshot Kgalagadi (Season 1). Retrieved from: http://lila.science/datasets/snapshot-kgalagadi\n",
        "\n",
        "Labeled Information Library of Alexandria (LILA BC) (n.d.) Snapshot Camdeboo (Season 1). Retrieved from: http://lila.science/datasets/snapshot-camdeboo\n",
        "\n",
        "Labeled Information Library of Alexandria (LILA BC) (n.d.) Snapshot Karoo (Season 1). Retrieved from: http://lila.science/datasets/snapshot-karoo\n"
      ]
    }
  ]
}






```python

```
