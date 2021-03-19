fi=open("del_ad_feature.csv")
line=fi.readline()
del_ads_feature_dict = dict()
while line:
    item = line.replace("\n","")
    del_ads_feature_dict[item] =1
    line = fi.readline()
fi=open("ads_feature.csv")
line = fi.readline()
while line:
    item = line.replace("\n","").split("\t")
    if item[0] not in del_ads_feature_dict:
        print(item[0]+"\t"+item[1])
    line = fi.readline()
