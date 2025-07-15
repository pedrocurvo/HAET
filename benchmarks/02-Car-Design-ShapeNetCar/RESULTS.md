# Results

# Experimental 

-cfd_model=ErwinTransolver500 \
    --data_dir data/shapenet_car/mlcfd_data/training_data \
    --save_dir data/shapenet_car/mlcfd_data/preprocessed_data \
    --weight 0.5 \
    --unified_pos 1 \
    --lr 0.00015 \
    --nb_epochs 500 \
    --slice_num 32

- Evaluation

rho_d:  0.9718691869186918
c_d:  0.023998820144204215
relative l2 error press: 0.094715156
relative l2 error velo: 0.02982628474603058
press: 5.9624087408962625
velo: [0.15017311 0.17917855 0.5358971 ] 0.33756116
time: 0.04204524755477905