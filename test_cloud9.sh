#!/bin/bash

sudo apt-get install -y ipython python-matplotlib python-sklearn python-pandas unzip

scp ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/kaggle_acts_of_pizza/*.zip .

for F in *.zip;
do
    unzip -x $F;
done

./my_model.py

ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk DONE"
