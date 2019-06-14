def datamanuplation(process,dictionary):
    list1=[]
    try:
        for sub in process:
            for key in sub.values():
                if(dictionary.get(key) != None):
                    list1.append(str(dictionary.get(key)))
                else:
                    list1.append(int(key))
        return list1            
    except Exception as e:
        return None
    
