#! /usr/bin/awk -f
BEGIN{
 FS=";"
 OFS=","
}
function oneHotVector(startIndex,lastIndex,oneIndex) {
  outstr=""
  for (i=startIndex;i<=lastIndex;i++){
    if(i==oneIndex) {
      outstr=outstr""OFS"1"
    } else {
      outstr=outstr""OFS"0"
    }
  }
  return(outstr)
}

$61!=""{
 if(int($61)==20){
  outstr=1
 } else {
  outstr=0
 }
 #print("myID"$54,"p2pID"$55,"cs"$56,"cho"$57,"chc"$58,"ce"$59,"cID"$60,"ex"$61)
 outstr=outstr""oneHotVector(-1,6,$25)
 print(outstr) > "train.csv"
}
