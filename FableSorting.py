import string

text = open('Fables.txt', 'r')

while True:
      
    # read by character
    char = text.read(1)          
    if not char: 
        break
          
    print(char)