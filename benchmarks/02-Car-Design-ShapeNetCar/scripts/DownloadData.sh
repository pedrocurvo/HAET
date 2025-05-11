# Instructions from https://github.com/ml-jku/UPT/blob/main/SETUP_DATA.md

cd $HOME/erwinpp/02-Car-Design-ShapeNetCar/data/shapenet_car

# Download the data
wget http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip

# Unzip the data
unzip mlcfd_data.zip

# Remove the zip file
rm mlcfd_data.zip

# Remove unnecessary files
rm -rf __MACOSX

cd mlcfd_data/training_data

# Unzip parameter files
tar -xvzf param0.tar.gz
tar -xvzf param1.tar.gz
tar -xvzf param2.tar.gz
tar -xvzf param3.tar.gz
tar -xvzf param4.tar.gz
tar -xvzf param5.tar.gz
tar -xvzf param6.tar.gz
tar -xvzf param7.tar.gz
tar -xvzf param8.tar.gz

# Remove tar files
rm param0.tar.gz
rm param1.tar.gz
rm param2.tar.gz
rm param3.tar.gz
rm param4.tar.gz
rm param5.tar.gz
rm param6.tar.gz
rm param7.tar.gz
rm param8.tar.gz

# Remove folders without quadpress_smpl.vtk
rm -rf ./param2/854bb96a96a4d1b338acbabdc1252e2f
rm -rf ./param2/85bb9748c3836e566f81b21e2305c824
rm -rf ./param5/9ec13da6190ab1a3dd141480e2c154d3
rm -rf ./param8/c5079a5b8d59220bc3fb0d224baae2a

# Preprocess the data
cd ../..
python preprocess.py --src /mlcfd_data/training_data --dst /mlcfd_data/preprocessed_data