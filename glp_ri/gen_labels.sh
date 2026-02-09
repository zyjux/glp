#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate glp
echo "Starting to generate training labels"
python ~/glp/glp_ri/generate_labels.py --filename train_labels.json --years 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2015 2016 2017 2018 2019
echo "Starting to generate validation labels"
python ~/glp/glp_ri/generate_labels.py --filename valid_labels.json --years 2005 2006 2007 2008 2009
echo "Starting to generate test labels"
python ~/glp/glp_ri/generate_labels.py --filename test_labels.json --years 2010 2011 2012 2013 2014
