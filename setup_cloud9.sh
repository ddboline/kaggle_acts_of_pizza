#!/bin/bash

sudo apt-get install -y ipython python-matplotlib python-sklearn python-pandas

scp ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/kaggle_pizza/*.zip .

for F in *.zip;
do
    unzip -x $F;
done
