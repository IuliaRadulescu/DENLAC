from __future__ import division
import sys

'''
Primeste ca parametru niste fisiere de tipul seturilor de date de test (float \t float \t int)
si le reindexeaza de la 0 -> fisierele noi au numele de tipul fisiervechi_changed.extensiefisiervechi
(poate primi orice numar de fisiere ca parametru)
'''

if __name__ == "__main__":

	list_of_files = sys.argv[1:]
	print(list_of_files)

	for filename in list_of_files:
		print(filename)
		changed_file_name = filename.split(".")[0]+"_changed."+filename.split(".")[1]
		with open(filename) as f:
			content = f.readlines()
		content = [l.strip() for l in content]
		for l in content:
			aux = l.split('\t')
			aux[2] = int(aux[2])
			aux[2] = str(aux[2]-1)
			changedl = '\t'.join(aux)
			with open(changed_file_name, 'a') as changed_file:
				changed_file.write(str(changedl)+'\n')
			changed_file.close()
