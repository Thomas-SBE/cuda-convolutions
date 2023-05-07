#!/bin/bash

echo -n "" > data

for i in {1..150}
do
    #build/blur | cut -d ' ' -f5 | tr '.' ',' >> data 
    build/b_shared | cut -d ' ' -f4 | tr '.' ',' >> data 
done