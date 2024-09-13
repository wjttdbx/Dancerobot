
def deal_with_language(s):
    global class_names
    lis = s.strip('[').strip(']').split(',')
    for word in lis:
        if word.strip().strip('\'') in types:
            if word.strip().strip('\'') =="biscuit":
                word='cookies'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="chip":
                word='potato wish'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="lays":
                word='potato chips'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="bread":
                word='bread'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="cookie":
                word='biscuit'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="handwash":
                word='Liquid soap'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="dishsoap":
                word='detergent'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="water":
                word='spring'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="coke":
                word='cola'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="orange juice":
                word='orange water'
                print('word is ', word)
                return word.strip().strip('\'')
            if word.strip().strip('\'') =="Shampoo":
                word='shampoo'
                print('word is ', word)
                return word.strip().strip('\'')



objects=[]
objects.append("object_name1")
objects.append("object_name2")
objects.append("object_name3")
print(objects)
if "object_name3" in objects:
    print(123)

if __name__ == "__main__":
    types = ['biscuit', 'chip', 'lays', 'bread', 'cookie', 'handwash', 'shampoo', 'dishsoap', 'water', 'sprite',
             'cola', 'orange juice', 'shampoo']  # 改为自己数据集的label
    word=deal_with_language("biscuit")
    print(word)
    # classes = ['cookies', 'coke', 'sprite', 'potato wish', 'potato chips', 'spring', 'shampoo', 'detergent',
    #            'orange water', 'bread', 'biscuit', 'Liquid soap']  # 改为自己数据集的label
