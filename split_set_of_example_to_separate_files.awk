FNR==1{
  split(FILENAME,fname,"[.x]")
  baseOutFileName = fname[1]"x"fname[2]
  prefix = fname[2]
  p=0
  outFileName = "partOfCsv_"prefix"/"baseOutFileName"_p"p".csv"
}
{
  if(FNR%100000==0) {
    p++
    outFileName = "partOfCsv_"prefix"/"baseOutFileName"_p"p".csv"
  }
  print($0) >> outFileName
}