import pickle



def preprocess_labels():
    charSet = {}
    val = 0
    with open("lines.txt") as infile:
        with open("raw_lines.txt", 'a') as outfile:
            for line in infile:
                lineList = (line.split())
                output = lineList[-1]
                outputlst = list(output.replace('|', ' '))
                write = lineList[0] + " <start>" + output +"<end>\n"
                for a in output:
                    if charSet.get(a, None) == None:
                        charSet[a] = val
                        val += 1
                outfile.write(write)
    print (charSet.keys())
    print (len(charSet.keys()))
    with open('encoding.pkl', 'wb') as f:
        pickle.dump(charSet, f, pickle.HIGHEST_PROTOCOL)