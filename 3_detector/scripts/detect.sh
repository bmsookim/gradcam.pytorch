#!/bin/bash

# subtypes = ['WBC_Basophil', 'WBC_Eosinophil', 'WBC_Lymphocyte', 'WBC_Lymphocyte_atypical', 'WBC_Monocyte', 'WBC_Neutrophil_Band', 'WBC_Neutrophil_Segmented', 'WBC_Smudge']

#for ((i=14; i<=27; i++)) do
for i in 8 9 16 17
do
    python detect_cell.py \
        --net_type resnet \
        --depth 50 \
        --windowSize 90 \
        --stepSize 50 \
        --testNumber $i
        #--subtype WBC_Lymphocyte
done
