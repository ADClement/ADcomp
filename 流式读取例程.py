from csv import DictWriter
import os
os.chdir(r'D:\tffile\比赛\preliminary_contest_data\data')

ix = 0

fo =  open('userFeature%s.csv' %ix, 'w')
headers = ['uid', 'age', 'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS', 'interest1', 'interest2',
	'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3',  'topic1', 'topic2', 'topic3', 'appIdInstall',
	'appIdAction', 'ct', 'os', 'carrier', 'house']
writer = DictWriter(fo, fieldnames=headers, lineterminator='\n')
writer.writeheader()

fi = open('userFeature.data', 'r')
for line in fi :
	line = line.replace('\n', '').split('|')
	userFeature_dict = {}
	for each in line:			
		each_list = each.split(' ')
		userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
		writer.writerow(userFeature_dict)
		ix = ix+1
		if ix % 400000==0:
			print(ix)
			#fo.close()
			fo = open('userFeature%s.csv' %ix, 'w')
			writer = DictWriter(fo, fieldnames=headers, lineterminator='\n')
			writer.writeheader()
fo.close()
fi.close()




    

