def judgeCircle(moves):
	res=False
	U_count=moves.count('U')    #cnts occurences of 'U'
	D_count=moves.count('D')    #cnts occurences of 'D'
	L_count=moves.count('L')    #cnts occurences of 'L'
	R_count=moves.count('R')    #cnts occurences of 'R'
	if U_count==D_count and L_count==R_count:   
		res=True
	else:
		res=False
		
	return res

path=input('Enter path:')  #user input
print(judgeCircle(path))
