def test():
	a=1
	b=2
	return a,b

print(test()) # (1, 2)

a,b=test()
print(a) # 1
print(b) # 2

print(test()[0]) # 1
print(test()[1]) # 2
