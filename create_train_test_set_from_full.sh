#! /bin/bash
if [ -f dataset/"full_"$1".csv" ] ; then
   head -n 178779 "dataset/full_"$1".csv" > "dataset/"$1".csv"
   number_of_lines=$(wc -l "dataset/full_"$1".csv" | cut -d" " -f1 )
   tail -n $(($number_of_lines-178779)) "dataset/full_"$1".csv" > "dataset/"$1"test.csv" 
fi
if [ -f dataset/"full_"$1".svmlight" ] ; then
   head -n 178778 "dataset/full_"$1".svmlight" > "dataset/"$1".svmlight"
   number_of_lines=$(wc -l "dataset/full_"$1".svmlight" | cut -d" " -f1 )
   tail -n $(($number_of_lines-178778)) "dataset/full_"$1".svmlight" > "dataset/"$1"test.svmlight"
fi
