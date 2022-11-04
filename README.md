# TensorsFeatureExtractionBathymetry
This is a MATLAB script demonstrating a deep learning-based geodetic bathymetry prediction using gravity gradient tensors as input signals. <br>
The MATLAB wrapper for the Generic Mapping Tools (GMT) is extensively used in this code. 
You only need to install GMT 6.3.0 or above, available at https://github.com/GenericMappingTools/gmt/releases. <br>
This demonstration was done on a computer with 64 GB RAM, i7-12700H CPU and RTX 3050Ti GPU. It has been tested on MATLAB R2020b and above. <br>

The datasets are grids of gravity gradient tensors (Txx.nc, Tyy.nc, Tzz,nc, Txy.nc, Txz.nc and Tyz.nc), and ship-borne depths (Sounding.mat).

It takes roughly 3 days to fully run this script. <br>
There is a possibility of running out of memory as the demonstration area (lon -1 ~ 60, lat -31 ~ 30) is very large. <br>
Access to a more powerful GPU is highly recommended. It is advisable to cut out a study area smaller than, or preferably half of, this one if your computer has a 32 GB RAM. <br>

There are further instructions as you run the code. <br>
Since the code is a livescript, you are advised to run it section-by-section.

This code supports a research article currently under review. I will provide the reference for the article immediately it is published.

By: Richard Fiifi ANNAN (richannan@outlook.com) <br>
    School of Land Science and Technology <br>
    China University of Geosciences (Beijing) <br>
    No. 29 Xueyuan Road, Haidian District, Beijing, China <br>
