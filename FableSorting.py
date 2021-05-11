import string

text = open('Fables.txt', 'r')
i = 0
c = 0
f = open("Fable" + str(c) + ".txt", "x") 

while True:
    # read by character
    char = text.read(1)
    if char == '\n':
        i += 1
    else:
        i = 0
    if i > 4:
        c += 1
        f.close
        f = open("Fable" + str(c) + ".txt", "x")           
    if not char: 
        break
          
    f.write(char)