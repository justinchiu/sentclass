#!/bin/bash

curl -O http://annotate-neighborhood.com/download/sentihood-train.json 
curl -O http://annotate-neighborhood.com/download/sentihood-dev.json 
curl -O http://annotate-neighborhood.com/download/sentihood-test.json 

curl -O https://raw.githubusercontent.com/uclmr/jack/master/data/sentihood/sentihood-train.json
curl -O https://raw.githubusercontent.com/uclmr/jack/master/data/sentihood/sentihood-dev.json
curl -O https://raw.githubusercontent.com/uclmr/jack/master/data/sentihood/sentihood-test.json
curl -O https://raw.githubusercontent.com/uclmr/jack/master/data/sentihood/single_jtr.json
